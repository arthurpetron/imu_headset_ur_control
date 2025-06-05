from ezmsg.core import Component, InputStream, OutputStream, Message
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import numpy as np
import asyncio

class UR5eMotionClient(Component):
    INPUT_JOINTS = InputStream(np.ndarray)
    OUTPUT_FEEDBACK = OutputStream(np.ndarray)

    def __init__(self, host: str = "192.168.0.100", frequency: float = 10.0):
        self.host = host
        self.frequency = frequency
        self.rtde_c = None
        self.rtde_r = None
        self._feedback_task = None

    async def initialize(self) -> None:
        self.rtde_c = RTDEControl(self.host)
        self.rtde_r = RTDEReceive(self.host)
        self._feedback_task = asyncio.create_task(self._feedback_loop())

    async def shutdown(self) -> None:
        if self._feedback_task:
            self._feedback_task.cancel()
        if self.rtde_c:
            self.rtde_c.stopScript()
        if self.rtde_r:
            self.rtde_r.disconnect()

    async def on_input(self, msg: Message) -> None:
        joints = msg.data
        if self.rtde_c:
            self.rtde_c.moveJ(joints.tolist(), speed=1.0, acceleration=1.2)

    async def _feedback_loop(self):
        while True:
            if self.rtde_r:
                feedback = self.rtde_r.getActualQ()
                await self.OUTPUT_FEEDBACK.send(np.array(feedback))
            await asyncio.sleep(1.0 / self.frequency)