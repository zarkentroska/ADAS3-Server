import os
import queue
import threading
import time

import cv2


class ThreadedVideoCapture:
    """
    Captura de video optimizada con reconexión segura.
    """

    def __init__(self, src):
        self.src = src
        self.successful_init = False
        self.stopped = False
        self.frame = None
        self.ret = False
        self.frame_id = 0
        self.init_time = time.time()
        self.last_frame_time = time.time()
        self.lock = threading.Lock()

        if os.name == "nt":
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        if not self.cap.isOpened():
            print("[VIDEO] Error: No se pudo abrir el stream.")
            return

        self.successful_init = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print("[VIDEO] Captura Low-Latency iniciada")

    def is_valid(self):
        return self.successful_init

    def _update(self):
        while not self.stopped:
            if self.cap.isOpened():
                grabbed = self.cap.grab()
                if grabbed:
                    self.ret, frame = self.cap.retrieve()
                    if self.ret and frame is not None:
                        with self.lock:
                            self.frame = frame
                            self.frame_id += 1
                            self.last_frame_time = time.time()
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.1)

    def read(self):
        if not self.successful_init:
            return False, None, -1

        with self.lock:
            if self.frame is not None:
                if time.time() - self.last_frame_time > 3.0:
                    return False, None, -1
                return True, self.frame.copy(), self.frame_id

            if time.time() - self.init_time < 5.0:
                return True, None, -1

            return False, None, -1

    def release(self):
        self.stopped = True
        if self.successful_init and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap.isOpened():
            self.cap.release()
        print("[VIDEO] Captura liberada")


class VideoConnectionManager:
    """
    Gestiona intentos asíncronos de conexión de video.
    """

    def __init__(self):
        self._attempts = []

    def schedule(self, target_url, force=False):
        if not force:
            for attempt in self._attempts:
                if (
                    attempt.get("url") == target_url
                    and attempt.get("thread")
                    and attempt["thread"].is_alive()
                ):
                    return

        result_queue = queue.Queue(maxsize=1)

        def worker(url, result_q):
            new_cap = None
            try:
                print(f"[VIDEO] Intentando conectar a {url} (async)...")
                new_cap = ThreadedVideoCapture(url)
                if not new_cap.is_valid():
                    new_cap.release()
                    new_cap = None
            except Exception as e:
                print(f"[VIDEO] Error al iniciar conexión: {e}")
                if new_cap:
                    new_cap.release()
                    new_cap = None
            finally:
                try:
                    result_q.put((url, new_cap), timeout=1)
                except queue.Full:
                    if new_cap:
                        new_cap.release()

        thread = threading.Thread(target=worker, args=(target_url, result_queue), daemon=True)
        self._attempts.append({"thread": thread, "queue": result_queue, "url": target_url})
        thread.start()

    def process_pending(self, current_cap, current_url):
        if not self._attempts:
            return current_cap, False

        new_cap_assigned = False
        remaining_attempts = []

        for attempt in self._attempts:
            result_q = attempt["queue"]
            try:
                result_url, candidate_cap = result_q.get_nowait()
            except queue.Empty:
                remaining_attempts.append(attempt)
                continue

            if (
                candidate_cap
                and candidate_cap.is_valid()
                and current_cap is None
                and result_url == current_url
                and not new_cap_assigned
            ):
                current_cap = candidate_cap
                new_cap_assigned = True
                print(f"[VIDEO] Conexión establecida con {result_url}")
            elif candidate_cap:
                candidate_cap.release()

        self._attempts = remaining_attempts
        return current_cap, new_cap_assigned
