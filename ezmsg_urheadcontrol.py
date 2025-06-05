# ezmsg_urheadcontrol.py
import numpy as np
from typing import Generator
from ezmsg.util.generator import consumer, Gen
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.pose import PoseStamped
from ur5e_kinematics import inverse_kinematics

@consumer
def head_to_pose() -> Generator[np.ndarray, tuple[PoseStamped, AxisArray], None]:
    controller = UR5eControlInterface()
    pose_msg = PoseStamped()
    joint_msg = AxisArray()

    while True:
        theta = yield (pose_msg, joint_msg)
        pos, _ = controller.update_from_head_orientation(theta)
        quat = controller.compute_tcp_orientation()

        pose_msg = PoseStamped(
            frame_id="base_link",
            timestamp=None,  # To be filled by ezmsg stream timing
            position=pos.tolist(),
            orientation=quat.tolist()
        )

        joints = inverse_kinematics(pos, quat)
        joint_msg = AxisArray(data=joints if joints is not None else np.zeros(6))


class UR5eHeadControl(Gen):
    def construct_generator(self):
        self.STATE.gen = head_to_pose()