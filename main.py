import asyncio
import numpy as np

from ezmsg.core import Graph, run
from ezmsg.util.rate import RateSource
from ezmsg.util.fn import FnComponent
from ezmsg.util.debuglog import DebugLog
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.pose import PoseStamped

from ezmsg_urheadcontrol import UR5eHeadControl
from ur5e_motion_client import UR5eMotionClient  # New generator-style motion client
from headset_imu_server import UDPIMUServerThread
from imu_state import IMUStateHistory
from simple_head_simulator import SimpleHeadSimulator

# Shared IMU state and simulation setup
imu_state = IMUStateHistory()
udp_server = UDPIMUServerThread(imu_state)
simulator = SimpleHeadSimulator(I=None, k=None, gamma=None)

def sample_head_orientation() -> np.ndarray:
    ts, acc, gyro = imu_state.get_latest()
    simulator.apply_imu(np.array(gyro), np.array(acc), timestamp=ts)
    return np.radians(simulator.get_orientation_euler())

# Components
rate = RateSource(rate_hz=20.0)
sampler = FnComponent()
sampler.set_function(sample_head_orientation)

head_control = UR5eHeadControl()
motion_client = UR5eMotionClient()
logger_tcp_pose = DebugLog(prefix="TCP Pose")
logger_feedback = DebugLog(prefix="Joint Feedback")

# Graph
graph = Graph()

graph.connect(rate.OUTPUT, sampler.INPUT)
graph.connect(sampler.OUTPUT, head_control.INPUT)

graph.connect(head_control.OUTPUT[0], logger_tcp_pose.INPUT)
graph.connect(head_control.OUTPUT[1], motion_client.INPUT)

graph.connect(motion_client.OUTPUT, logger_feedback.INPUT)

# Run loop
async def main():
    udp_server.start()
    try:
        await run(graph)
    finally:
        udp_server.stop()
        udp_server.join()

if __name__ == "__main__":
    asyncio.run(main())