import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from reskin_sensor import ReSkinProcess

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

    # labels of plot
    labels = ["t3", "Bx3", "By3", "Bz3"]

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
                # Append new data
                data[0].append(values[12])  # t3
                data[1].append(values[13])  # Bx3
                data[2].append(values[14])  # By3
                data[3].append(values[15])  # Bz3
                
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
                lines[0].set_data(x, data[0])  # renew t3
                lines[1].set_data(x, data[1])  # renew Bx3
                lines[2].set_data(x, data[2])  # renew By3
                lines[3].set_data(x, data[3])  # renew Bz3

        return lines

    # Start sensor stream
    sensor_stream.start()
    time.sleep(0.1)

    # visualize
    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100, save_count=50, cache_frame_data=False)

    plt.xlabel('Sample Number')
    plt.show()

if __name__ == '__main__':
    main()