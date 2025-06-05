import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import pybullet as p
import pybullet_data
import numpy as np
import time

class UR5ePyBulletVisualizer(Node):
    def __init__(self):
        super().__init__('ur5e_pybullet_visualizer')
        self.subscription = self.create_subscription(
            Pose,
            '/ur5e/tcp_command',
            self.listener_callback,
            10)

        # Start PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane_id = p.loadURDF('plane.urdf')
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        urdf_path = "urdf/ur5e.urdf"  # Make sure this path points to the UR5e URDF
        self.robot_id = p.loadURDF(urdf_path, start_pos, start_orientation, useFixedBase=True)

        self.end_effector_index = 7  # UR5e TCP link index (adjust if needed)
        self.joint_indices = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] == p.JOINT_REVOLUTE]

        self.timer = self.create_timer(1.0 / 60.0, self.step_simulation)
        self.last_target_pose = None

    def listener_callback(self, msg):
        self.last_target_pose = msg

    def step_simulation(self):
        if self.last_target_pose is not None:
            pos = [
                self.last_target_pose.position.x,
                self.last_target_pose.position.y,
                self.last_target_pose.position.z
            ]
            orn = [
                self.last_target_pose.orientation.x,
                self.last_target_pose.orientation.y,
                self.last_target_pose.orientation.z,
                self.last_target_pose.orientation.w
            ]

            joint_angles = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                pos,
                orn,
                maxNumIterations=200,
                residualThreshold=1e-4
            )

            for i, joint_index in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=joint_angles[i],
                    force=500)

        p.stepSimulation()


def main(args=None):
    rclpy.init(args=args)
    node = UR5ePyBulletVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    p.disconnect()


if __name__ == '__main__':
    main()