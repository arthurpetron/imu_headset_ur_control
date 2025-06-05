# ur5e_motion_client.py
import numpy as np
import time
from typing import Generator

from ezmsg.util.generator import consumer, Gen
from ezmsg.util.messages.axisarray import AxisArray

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

@consumer
def ur5e_motion(host: str = "192.168.0.100", hz: float = 10.0) -> Generator[AxisArray, AxisArray, None]:
    # Connection setup
    rtde_c = RTDEControl(host)
    rtde_r = RTDEReceive(host)

    try:
        last_feedback_time = time.time()
        interval = 1.0 / hz

        joint_cmd = np.zeros(6)  # default dummy input
        feedback = AxisArray(data=np.zeros(6))

        while True:
            # receive command
            joint_cmd = yield feedback
            rtde_c.moveJ(joint_cmd.tolist(), speed=1.0, acceleration=1.2)

            # feedback update
            now = time.time()
            if now - last_feedback_time >= interval:
                last_feedback_time = now
                feedback = AxisArray(data=np.array(rtde_r.getActualQ()))
    finally:
        rtde_c.stopScript()
        rtde_r.disconnect()

class UR5eMotionClient(Gen):
    def construct_generator(self):
        self.STATE.gen = ur5e_motion()