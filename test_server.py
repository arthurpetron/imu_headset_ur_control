from neuroplastic_tribeca import IMUStateHistory
from neuroplastic_tribeca import UDPIMUServerThread

if __name__ == "__main__":
    imu_state = IMUStateHistory()
    server_thread = UDPIMUServerThread(imu_state)
    server_thread.start()

    try:
        while True:
            latest = imu_state.get_latest()
            print(f"Latest IMU: {latest}")
            time.sleep(1)
    except KeyboardInterrupt:
        server_thread.stop()
        server_thread.join()