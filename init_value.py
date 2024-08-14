import numpy as np
import time
from reskin_sensor import ReSkinProcess
import argparse

def initialize_sensor(sensor_stream, duration=20, sampling_rate=100):
    """
    Initialize the sensor by collecting data for a specified duration and calculating the average values.
    
    Args:
    - sensor_stream: The sensor stream object to collect data from.
    - duration: The duration (in seconds) to collect data for initialization.
    - sampling_rate: The rate (in Hz) at which to sample data.
    
    Returns:
    - init_values: A list of initial average values for t1, Bx1, By1, and Bz1.
    """
    collected_data = {i: [] for i in range(0,20)}  # Dictionary to hold data for t1, Bx1, By1, and Bz1

    # Calculate the total number of samples to collect
    total_samples = duration * sampling_rate

    # Collect data
    for _ in range(total_samples):
        if sensor_stream.is_alive():
            sample = sensor_stream.get_data(num_samples=1)[0]
            values = sample.data
            if len(values) == 20:
                for i in range(0, 20):
                    collected_data[i].append(values[i])
        time.sleep(1 / sampling_rate)

    # Apply filtering (sliding window average)
    window_size = 5
    filtered_data = {i: np.convolve(collected_data[i], np.ones(window_size) / window_size, mode='valid') for i in collected_data}

    # Calculate the average of the filtered data
    init_values = [np.mean(filtered_data[i]) for i in range(0, 20)]

    return init_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test code to run a ReSkin streaming process in the background. Allows data to be collected without code blocking"
    )
    # fmt: off
    parser.add_argument("-p", "--port", type=str, help="port to which the microcontroller is connected", required=True,)
    parser.add_argument("-b", "--baudrate", type=str, help="baudrate at which the microcontroller is streaming data", default=115200,)
    parser.add_argument("-n", "--num_mags", type=int, help="number of magnetometers on the sensor board", default=5,)
    parser.add_argument("-tf", "--temp_filtered", action="store_true", help="flag to filter temperature from sensor output",)
    # fmt: on
    args = parser.parse_args()

    # Create sensor stream
    sensor_stream = ReSkinProcess(
    num_mags=args.num_mags,
    port=args.port,
    baudrate=args.baudrate,
    burst_mode=True,
    device_id=1,
    temp_filtered=args.temp_filtered,
    )
    # Start sensor stream
    sensor_stream.start()
    time.sleep(0.1)

    # Example usage (assuming sensor_stream is already defined and started):
    init_values = initialize_sensor(sensor_stream)
    print("Initial values:", init_values)
