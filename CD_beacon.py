import asyncio
 
from bleak import BleakScanner
 
import matplotlib.pyplot as plt
 
import numpy as np
 
 
class KalmanFilter:
 
    def __init__(self, process_noise=0.005, measurement_noise=20):
 
        self.initialized = False
 
        self.process_noise = process_noise
 
        self.measurement_noise = measurement_noise
 
        self.predicted_rssi = 0
 
        self.error_covariance = 0
 
    def filter(self, rssi):
 
        if not self.initialized:
 
            self.initialized = True
 
            prior_rssi = rssi
 
            prior_error_covariance = 1
 
        else:
 
            prior_rssi = self.predicted_rssi
 
            prior_error_covariance = self.error_covariance + self.process_noise
 
        kalman_gain = prior_error_covariance / (prior_error_covariance + self.measurement_noise)
 
        self.predicted_rssi = prior_rssi + kalman_gain * (rssi - prior_rssi)
 
        self.error_covariance = (1 - kalman_gain) * prior_error_covariance
 
        return self.predicted_rssi
 
 
def rssi_to_distance(rssi, tx_power=-59, n=2):
 
    return 10 ** ((tx_power - rssi) / (10 * n))
 
def triangulate(beacon_data, beacon_positions):
    if len(beacon_data) < 3:
        return None  # 삼변측량을 위해 최소 세 개의 비콘 필요

    distances = [rssi_to_distance(beacon['rssi']) for beacon in beacon_data]
    weights = [1/d for d in distances]  # 각 거리의 역수를 가중치로 사용
    x, y = np.average([pos[0] for pos in beacon_positions], weights=weights), np.average([pos[1] for pos in beacon_positions], weights=weights)
    return x, y

 
def display_points_on_graph(positions):
 
    plt.figure(figsize=(10, 6))
 
    for pos in positions:
 
        plt.scatter(*pos, color='red')
 
        # plt.text(pos[0], pos[1], f' ({pos[0]}, {pos[1]})', color='blue', fontsize=12)
 
    plt.xlim(0, 15)
 
    plt.ylim(0, 10)
 
    plt.title('Location Plot')
 
    plt.xlabel('X Coordinate')
 
    plt.ylabel('Y Coordinate')
 
    plt.grid(True)
 
    plt.show()

 
async def scan_and_display():
    scanner = BleakScanner()
    kalman_filters = {}
    beacon_positions = {
        'C3:00:00:1C:6E:69': (0, 5.34),
        'C3:00:00:1C:6E:60': (7.2, 0),
        'C3:00:00:1C:6E:6F': (0, 0),
        'C3:00:00:1C:6E:6A': (3.6, 5.34),
        'C3:00:00:1C:6E:5E': (7.2, 5.34),
        'C3:00:00:1C:6E:5F': (3.6, 0)
    }
    collected_positions = []
 
    while True:
        devices = await scanner.discover()
        beacon_data = []
 
        for device in devices:
            if device.name == "MBeacon" and device.address in beacon_positions:
                if device.address not in kalman_filters:
                    kalman_filters[device.address] = KalmanFilter()
 
                filtered_rssi = kalman_filters[device.address].filter(device.rssi)
                beacon_data.append({'rssi': filtered_rssi, 'address': device.address})
 
                if len(beacon_data) >= 3:
                    positions = [beacon_positions[data['address']] for data in beacon_data]
                    estimated_position = triangulate(beacon_data, positions)
                    if estimated_position:
                        collected_positions.append(estimated_position)
                        print(f"Collected position: {estimated_position}")
                        if len(collected_positions) >= 10:
                            display_points_on_graph(collected_positions)
                            collected_positions = []  # Reset the list after displaying
                    break
 
        await asyncio.sleep(1)  # Reduce sleep time to update more frequently
 
def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(scan_and_display())
 
if __name__ == "__main__":
    main()