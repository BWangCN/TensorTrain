import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from reskin_sensor import ReSkinProcess
from init_value import initialize_sensor  # 引入初始化函数

def main():
    parser = argparse.ArgumentParser(
        description="Real-time visualization of ReSkin sensor data"
    )
    parser.add_argument("-p", "--port", type=str, help="Port to which the microcontroller is connected", required=True)
    parser.add_argument("-b", "--baudrate", type=int, help="Baudrate at which the microcontroller is streaming data", default=115200)
    parser.add_argument("-n", "--num_mags", type=int, help="Number of magnetometers on the sensor board", default=5)
    parser.add_argument("-tf", "--temp_filtered", action="store_true", help="Flag to filter temperature from sensor output")
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

    # Start sensor stream and initialize sensor
    sensor_stream.start()
    time.sleep(0.1)
    init_values = initialize_sensor(sensor_stream)  # 获取初始值
    print("Initial values:", init_values)

    # labels of plot
    labels = ["t0", "Bx0", "By0", "Bz0"]

    # initialize plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    fig.subplots_adjust(hspace=0.4)
    
    lines = []
    for i, label in enumerate(labels):
        line, = axs[i].plot([], [], label=label)
        lines.append(line)
        axs[i].set_xlim(0, 100)
        axs[i].set_ylim(-3000, 3000)
        axs[i].set_title(label)
        axs[i].legend(loc='upper right')

    # store data
    data = [[] for _ in range(4)]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        if sensor_stream.is_alive():
            sample = sensor_stream.get_data(num_samples=1)[0]  # acquire a sample
            values = sample.data
            if len(values) == 20:
                # Subtract initial values
                t0 = values[0] - init_values[0]
                Bx0 = values[1] - init_values[1]
                By0 = values[2] - init_values[2]
                Bz0 = values[3] - init_values[3]

                # Append new data
                data[0].append(t0)  # t0
                data[1].append(Bx0)  # Bx0
                data[2].append(By0)  # By0
                data[3].append(Bz0)  # Bz0
                
                # Ensure data lists do not exceed 100 points
                if len(data[0]) > 100:
                    data[0].pop(0)
                if len(data[1]) > 100:
                    data[1].pop(0)
                if len(data[2]) > 100:
                    data[2].pop(0)
                if len(data[3]) > 100:
                    data[3].pop(0)

                # Create x values based on the length of data[0]
                x = list(range(len(data[0])))

                # Update the plot lines with the new data
                lines[0].set_data(x, data[0])  # renew t0
                lines[1].set_data(x, data[1])  # renew Bx0
                lines[2].set_data(x, data[2])  # renew By0
                lines[3].set_data(x, data[3])  # renew Bz0

        return lines

    # visualize
    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100, save_count=50, cache_frame_data=False)

    plt.xlabel('Sample Number')
    plt.show()

if __name__ == '__main__':
    main()

