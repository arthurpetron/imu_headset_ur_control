import time
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

class UR5eMotionClient:
    def __init__(self, host: str = "192.168.0.100", hz: float = 10.0):
        self.host = host
        self.hz = hz
        self.rtde_c = None
        self.rtde_r = None
        self.interval = 1.0 / hz
        self.running = False

    def connect(self):
        self.rtde_c = RTDEControl(self.host)
        self.rtde_r = RTDEReceive(self.host)
        self.running = True
        print(f"[UR5e] Connected to {self.host}")

    def disconnect(self):
        if self.rtde_c:
            self.rtde_c.stopScript()
        if self.rtde_r:
            self.rtde_r.disconnect()
        self.running = False
        print("[UR5e] Disconnected.")

    def send_joint_command(self, joint_positions: np.ndarray, speed=1.0, acceleration=1.2):
        if self.rtde_c:
            self.rtde_c.moveJ(joint_positions.tolist(), speed, acceleration)

    def get_feedback(self) -> np.ndarray:
        if self.rtde_r:
            return np.array(self.rtde_r.getActualQ())
        return np.zeros(6)

    def run_loop(self, command_generator):
        """
        command_generator: yields np.ndarray joint command
        """
        self.connect()
        try:
            while self.running:
                joints = next(command_generator)
                self.send_joint_command(joints)
                feedback = self.get_feedback()
                print(f"Feedback: {np.round(feedback, 3)}")
                time.sleep(self.interval)
        except StopIteration:
            print("[UR5e] Command stream finished.")
        except KeyboardInterrupt:
            print("[UR5e] Interrupted by user.")
        finally:
            self.disconnect()


# if __name__ == "__main__":
#     client = UR5eMotionClient(host="192.168.0.100", hz=10.0)