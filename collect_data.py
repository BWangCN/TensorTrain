import csv
import time
import numpy as np
from reskin_sensor import ReSkinProcess
import argparse

def initialize_sensor(sensor_stream, duration=20, sampling_rate=500):
    """
    Initialize the sensor by collecting data for a specified duration and calculating the average values.
    
    Args:
    - sensor_stream: The sensor stream object to collect data from.
    - duration: The duration (in seconds) to collect data for initialization.
    - sampling_rate: The rate (in Hz) at which to sample data.
    
    Returns:
    - init_values: A list of initial average values for t0, Bx0, By0, Bz0, ..., t4, Bx4, By4, Bz4.
    """
    collected_data = {i: [] for i in range(20)}  # Dictionary to hold data for all sensors

    total_samples = duration * sampling_rate

    for _ in range(total_samples):
        if sensor_stream.is_alive():
            sample = sensor_stream.get_data(num_samples=1)[0]
            values = sample.data
            if len(values) == 20:
                for i in range(20):
                    collected_data[i].append(values[i])
        time.sleep(1 / sampling_rate)

    window_size = 5
    filtered_data = {i: np.convolve(collected_data[i], np.ones(window_size) / window_size, mode='valid') for i in collected_data}

    init_values = [np.mean(filtered_data[i]) for i in range(20)]

    return init_values

def collect_data(sensor_stream, init_values, label, duration=10, sampling_rate=500):
    """
    Collect sensor data and subtract the initial values.
    
    Args:
    - sensor_stream: The sensor stream object to collect data from.
    - init_values: The initial values to subtract from the collected data.
    - label: The label to assign to the collected data.
    - duration: Duration of data collection in seconds.
    - sampling_rate: Sampling rate in Hz.
    
    Returns:
    - collected_data: A list of collected and adjusted data points with the label.
    """
    total_samples = duration * sampling_rate
    collected_data = []

    for _ in range(total_samples):
        if sensor_stream.is_alive():
            sample = sensor_stream.get_data(num_samples=1)[0]
            values = sample.data
            if len(values) == 20:
                adjusted_values = [values[i] - init_values[i] for i in range(20)]
                sensor_values = adjusted_values[1::4] + adjusted_values[2::4] + adjusted_values[3::4]
                sensor_values.append(label)
                collected_data.append(sensor_values)
        time.sleep(1 / sampling_rate)

    return collected_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect sensor data and save to a CSV file.")
    parser.add_argument("-p", "--port", type=str, help="Port to which the microcontroller is connected", required=True)
    parser.add_argument("-b", "--baudrate", type=int, help="Baudrate at which the microcontroller is streaming data", default=115200)
    parser.add_argument("-n", "--num_mags", type=int, help="Number of magnetometers on the sensor board", default=5)
    parser.add_argument("-tf", "--temp_filtered", action="store_true", help="Flag to filter temperature from sensor output")
    parser.add_argument("-o", "--output_file", type=str, help="Output CSV file path", required=True)
    args = parser.parse_args()

    sensor_stream = ReSkinProcess(
        num_mags=args.num_mags,
        port=args.port,
        baudrate=args.baudrate,
        burst_mode=True,
        device_id=1,
        temp_filtered=args.temp_filtered,
    )
    sensor_stream.start()
    time.sleep(0.1)

    init_values = initialize_sensor(sensor_stream)
    print("Initial values:", init_values)

    all_data = []

    while True:
        label = int(input("Enter the label for the sample (0: No press, 1: Top, 2: Left, 3: Right, 4: End collection): "))
        if label == 4:
            break
        elif label in [0, 1, 2, 3]:
            print("Waiting for 1 seconds...")
            time.sleep(1)
            print("Starting data collection...")
            data = collect_data(sensor_stream, init_values, label)
            all_data.extend(data)
            print(f"Data collection for label {label} completed.")
        else:
            print("Invalid input, please enter again.")

    print("Data collection ended, saving to CSV file...")
    with open(args.output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Bx0', 'By0', 'Bz0', 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 'Bx3', 'By3', 'Bz3', 'Bx4', 'By4', 'Bz4', 'label']
        writer.writerow(header)
        writer.writerows(all_data)
    print(f"Data has been saved to {args.output_file}")

    sensor_stream.stop()

