import asyncio
import numpy as np

from ezmsg.core import Graph, run
from ezmsg.util.rate import RateSource
from ezmsg.util.fn import FnComponent
from ezmsg.util.messages.pose import PoseStamped

from headset_imu_server import UDPIMUServerThread
from imu_state import IMUStateHistory
from simple_head_simulator import SimpleHeadSimulator
from ur5e_control_interface import UR5eHeadControl

# 1. IMU shared state
imu_state = IMUStateHistory()
udp_server = UDPIMUServerThread(imu_state)
simulator = SimpleHeadSimulator(I=None, k=None, gamma=None)

# 2. Define callable that updates simulator and returns orientation
def sample_head_orientation() -> np.ndarray:
    ts, acc, gyro = imu_state.get_latest()
    simulator.apply_imu(np.array(gyro), np.array(acc), timestamp=ts)
    return np.radians(simulator.get_orientation_euler())

# 3. Build ezmsg graph
rate = RateSource(rate_hz=20.0)
sampler = FnComponent()
ur5e = UR5eHeadControl()

sampler.set_function(sample_head_orientation)

graph = Graph()
graph.connect(rate.OUTPUT, sampler.INPUT)
graph.connect(sampler.OUTPUT, ur5e.INPUT_ORIENTATION)

graph.export(ur5e.OUTPUT_TCP_POSE, "tcp_pose")
graph.export(ur5e.OUTPUT_JOINTS, "joint_angles")

# 4. Run
async def main():
    udp_server.start()
    try:
        await run(graph)
    finally:
        udp_server.stop()
        udp_server.join()

    motion_client = UR5eMotionClient(host='192.168.0.100')

    graph.connect(ur5e.OUTPUT_JOINTS, motion_client.INPUT_JOINTS)
    graph.export(motion_client.OUTPUT_FEEDBACK, "joint_feedback")

if __name__ == "__main__":
    asyncio.run(main())