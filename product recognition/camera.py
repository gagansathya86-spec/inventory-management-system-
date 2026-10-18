import cv2
import threading
import time
import numpy as np

class ThreadedCamera:
    """
    High-performance threaded camera stream that supports:
    - Built-in laptop webcam (device index 0)
    - Connected phone camera via DirectShow/virtual camera (device index 1, 2)
    - Phone camera / Gradio stream via network URL (http://, https://, rtsp://)
    Avoids OpenCV buffer lag by continuously dropping stale frames and caching the latest one.
    """
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_frame_time = 0.0
        self.is_synthetic = False
        self.is_connected_flag = False

        self._init_camera()

    def _is_network_stream(self):
        return isinstance(self.src, str) and (
            self.src.startswith("http://") or
            self.src.startswith("https://") or
            self.src.startswith("rtsp://")
        )

    def _init_camera(self):
        # 1. Check if source is a network/Gradio stream URL
        if self._is_network_stream():
            print(f"[Camera] Connecting to phone/network camera stream at: {self.src}...")
            try:
                self.cap = cv2.VideoCapture(self.src)
            except Exception as e:
                print(f"[Camera] Failed to open network stream: {e}")
                self.cap = None
        else:
            # Local DirectShow device index
            try:
                dev_idx = int(self.src)
            except (ValueError, TypeError):
                dev_idx = 0

            print(f"[Camera] Initializing local camera index {dev_idx} with DirectShow...")
            try:
                self.cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
            except Exception:
                self.cap = cv2.VideoCapture(dev_idx)

            if not self.cap or not self.cap.isOpened():
                print(f"[Camera] DirectShow failed, trying default backend for {dev_idx}...")
                self.cap = cv2.VideoCapture(dev_idx)

        # 2. Check if camera is successfully opened
        if self.cap is not None and self.cap.isOpened():
            if not self._is_network_stream():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest_frame = frame
                self.is_synthetic = False
                self.is_connected_flag = True
                print(f"[Camera] Connected successfully! Source: {self.src} ({frame.shape[1]}x{frame.shape[0]})")
                return
            else:
                print(f"[Camera] Warning: Camera {self.src} opened but failed to read initial frame.")

        # Camera failed / unavailable
        print(f"[Camera] Camera {self.src} is not accessible. Setting status to DISCONNECTED.")
        self.is_connected_flag = False
        self.is_synthetic = True
        self.latest_frame = self._generate_disconnected_frame()

    def _generate_disconnected_frame(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (20, 20, 28)
        label = f"CAMERA DISCONNECTED ({self.src})"
        cv2.putText(
            img, label, (30, self.height // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 70, 240), 2
        )
        cv2.putText(
            img, "Please check connection in Camera Settings", (30, self.height // 2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 170), 1
        )
        return img

    def is_connected(self):
        """Returns True if the camera is genuinely streaming real frames."""
        return self.is_connected_flag and not self.is_synthetic

    @classmethod
    def test_source(cls, candidate_src, timeout_sec=2.5):
        """
        Probes a candidate camera device index or network stream URL.
        Returns (success: bool, width: int, height: int, message: str)
        """
        if isinstance(candidate_src, str) and (
            candidate_src.startswith("http://") or
            candidate_src.startswith("https://") or
            candidate_src.startswith("rtsp://")
        ):
            cap = cv2.VideoCapture(candidate_src)
        else:
            try:
                dev_idx = int(candidate_src)
            except (ValueError, TypeError):
                dev_idx = 0
            cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(dev_idx)

        if not cap.isOpened():
            return False, 0, 0, f"Unable to open camera source '{candidate_src}'."

        start_time = time.time()
        ret, frame = False, None
        while time.time() - start_time < timeout_sec:
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.05)

        cap.release()

        if ret and frame is not None:
            h, w = frame.shape[:2]
            return True, w, h, f"Camera verified active ({w}x{h})."
        else:
            return False, 0, 0, f"Camera '{candidate_src}' opened but returned no frames within {timeout_sec}s."

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        consecutive_failures = 0
        while self.running:
            if not self.is_synthetic and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0
                    self.is_connected_flag = True
                    with self.lock:
                        self.latest_frame = frame
                        self.last_frame_time = time.time()
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        # Stream dropped
                        self.is_connected_flag = False
                        self.is_synthetic = True
                        with self.lock:
                            self.latest_frame = self._generate_disconnected_frame()
                    time.sleep(0.01)
            else:
                # Disconnected animation
                with self.lock:
                    self.latest_frame = self._generate_disconnected_frame()
                    self.last_frame_time = time.time()
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected_flag = False
        print(f"[Camera] Stream stopped and released for source {self.src}.")
