import threading
from collections import deque
from typing import List, Tuple
import time

class IMUStateHistory:
    def __init__(self, maxlen: int = 1000):
        self.lock = threading.Lock()
        self.history: deque[Tuple[float, List[float], List[float]]] = deque(maxlen=maxlen)
        # Each entry: (timestamp, acc_xyz, gyro_xyz)

    def add(self, acc: List[float], gyro: List[float], timestamp: float = None):
        with self.lock:
            if timestamp is None:
                timestamp = time.time()
            self.history.append((timestamp, acc, gyro))

    def get_latest(self) -> Tuple[float, List[float], List[float]]:
        with self.lock:
            return self.history[-1] if self.history else (0.0, [0.0]*3, [0.0]*3)
        
    def get_last_n(self, n: int) -> List[Tuple[float, List[float], List[float]]]:
        with self.lock:
            return list(self.history)[-n:] if n <= len(self.history) else list(self.history)

    def get_all(self) -> List[Tuple[float, List[float], List[float]]]:
        with self.lock:
            return list(self.history)