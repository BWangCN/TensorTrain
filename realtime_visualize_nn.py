import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import load
from reskin_sensor import ReSkinProcess
from init_value import initialize_sensor
from tensorflow.keras.models import load_model

def main():
    parser = argparse.ArgumentParser(
        description="Real-time visualization and prediction of ReSkin sensor data"
    )
    parser.add_argument("-p", "--port", type=str, help="Port to which the microcontroller is connected", required=True)
    parser.add_argument("-b", "--baudrate", type=int, help="Baudrate at which the microcontroller is streaming data", default=115200)
    parser.add_argument("-n", "--num_mags", type=int, help="Number of magnetometers on the sensor board", default=5)
    parser.add_argument("-tf", "--temp_filtered", action="store_true", help="Flag to filter temperature from sensor output")
    args = parser.parse_args()

    # Load the trained neural network model and scaler
    nn_model = load_model('nn_prediction_model10.keras')
    scaler = load('scaler.joblib')

    # Create sensor stream
    sensor_stream = ReSkinProcess(
        num_mags=args.num_mags,
        port=args.port,
        baudrate=args.baudrate,
        burst_mode=True,
        device_id=1,
        temp_filtered=args.temp_filtered,
    )

    # Start sensor stream and initialize sensor
    sensor_stream.start()
    time.sleep(0.1)
    init_values = initialize_sensor(sensor_stream)  # 获取初始值
    print("Initial values:", init_values)

    # labels of plot
    labels = ["Bx0", "By0", "Bz0", "Bx1", "By1", "Bz1", "Bx2", "By2", "Bz2", "Bx3", "By3", "Bz3", "Bx4", "By4", "Bz4"]

    # initialize plots
    fig, axs = plt.subplots(6, 3, figsize=(15, 24), sharex=True)
    fig.subplots_adjust(hspace=0.4)

    lines = []
    for i in range(5):
        for j, axis in enumerate(["Bx", "By", "Bz"]):
            idx = i * 3 + j
            line, = axs[i, j].plot([], [], label=labels[idx])
            lines.append(line)
            axs[i, j].set_xlim(0, 100)
            axs[i, j].set_ylim(-1000, 1000)
            axs[i, j].set_title(labels[idx])
            axs[i, j].legend(loc='upper right')

    # Add a subplot for displaying predictions
    prediction_text = axs[5, 1].text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', fontsize=15)
    axs[5, 1].axis('off')  # Hide the axis

    # store data
    data = [[] for _ in range(15)]

    def init():
        for line in lines:
            line.set_data([], [])
        prediction_text.set_text('')
        return lines + [prediction_text]

    def update(frame):
        if sensor_stream.is_alive():
            sample = sensor_stream.get_data(num_samples=1)[0]  # acquire a sample
            values = sample.data
            if len(values) == 20:
                adjusted_values = [values[i] - init_values[i] for i in range(20)]
                sensor_values = adjusted_values[1::4] + adjusted_values[2::4] + adjusted_values[3::4]
                sensor_values = np.array(sensor_values).reshape(1, -1)

                # 判断最大绝对值是否超过150
                if np.max(np.abs(sensor_values)) < 150:
                    prediction_text.set_text("Current press location: No press")
                else:
                    # Standardize the data
                    sensor_values_scaled = scaler.transform(sensor_values)

                    # Predict the label using the neural network model
                    probabilities = nn_model.predict(sensor_values_scaled)
                    label = np.argmax(probabilities, axis=1)[0]

                    # Display the result
                    label_names = {0: 'No press', 1: 'Top', 2: 'Left', 3: 'Right'}
                    prediction_text.set_text(f"Current press location: {label_names[label]}")

                for i in range(5):
                    # Subtract initial values for Bx, By, Bz
                    Bx = values[i * 4 + 1] - init_values[i * 4 + 1]
                    By = values[i * 4 + 2] - init_values[i * 4 + 2]
                    Bz = values[i * 4 + 3] - init_values[i * 4 + 3]

                    # Append new data
                    data[i * 3].append(Bx)  # Bx
                    data[i * 3 + 1].append(By)  # By
                    data[i * 3 + 2].append(Bz)  # Bz

                    # Ensure data lists do not exceed 100 points
                    if len(data[i * 3]) > 100:
                        data[i * 3].pop(0)
                    if len(data[i * 3 + 1]) > 100:
                        data[i * 3 + 1].pop(0)
                    if len(data[i * 3 + 2]) > 100:
                        data[i * 3 + 2].pop(0)

                    # Create x values based on the length of data[0]
                    x = list(range(len(data[i * 3])))

                    # Update the plot lines with the new data
                    lines[i * 3].set_data(x, data[i * 3])  # renew Bx
                    lines[i * 3 + 1].set_data(x, data[i * 3 + 1])  # renew By
                    lines[i * 3 + 2].set_data(x, data[i * 3 + 2])  # renew Bz

        return lines + [prediction_text]

    # visualize
    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=10, save_count=50, cache_frame_data=False)

    plt.xlabel('Sample Number')
    plt.show()

if __name__ == '__main__':
    main()


