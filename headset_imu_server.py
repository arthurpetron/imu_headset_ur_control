import socket
import threading
import struct
from neuroplastic_tribeca import IMUStateHistory  

class UDPIMUServerThread(threading.Thread):
    def __init__(self, imu_state: IMUStateHistory, port: int = 8000):
        super().__init__(daemon=True)
        self.imu_state = imu_state
        self.port = port
        self._stop_event = threading.Event()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", self.port))
        print(f"[UDP] Listening for IMU data on port {self.port}...")

        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(1024)
                if len(data) == 68:
                    self._handle_packet(data)
                else:
                    print(f"[UDP] Received packet of unexpected size: {len(data)} bytes")
            except Exception as e:
                print(f"[UDP Error] {e}")

        sock.close()

    def _handle_packet(self, data: bytes):
        try:
            # Unpack 17 float32 values from the 68-byte payload
            values = struct.unpack('<17f', data)
            # Extract accelerometer and gyroscope data
            acc = list(values[8:11])  # Channels 9-11
            gyro = list(values[11:14])  # Channels 12-14
            # Extract timestamp from channel 17
            timestamp = values[16]
            self.imu_state.add(acc=acc, gyro=gyro, timestamp=timestamp)
        except Exception as e:
            print(f"[Parse Error] Could not parse packet: {e}")

    def stop(self):
        self._stop_event.set()