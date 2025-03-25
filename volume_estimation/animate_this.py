import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

def main(csv_path: str) -> None:
    # Read CSV into a dataframe
    df: pd.DataFrame = pd.read_csv(csv_path)
    # Sort the dataframe by timestamp to maintain the order of frames
    df.sort_values('timestamp', inplace=True)

    # Extract unique timestamps and track IDs
    timestamps = sorted(df['timestamp'].unique())
    track_ids = df['track_id'].unique()

    # Create the figure and axis for plotting
    fig, ax = plt.subplots()

    # Create a dictionary to hold the line objects for each track
    lines = {track_id: ax.plot([], [], marker='o', label=f'Track {track_id}')[0] for track_id in track_ids}

    def init() -> List:
        """Initialize the plot with proper limits."""
        margin: float = 10.0  # Adjust margin if needed
        ax.set_xlim(df['center_x'].min() - margin, df['center_x'].max() + margin)
        ax.set_ylim(df['center_y'].min() - margin, df['center_y'].max() + margin)
        return list(lines.values())

    def update(frame: int) -> List:
        """Update the plot for the current frame."""
        current_time = timestamps[frame]
        # Get all data points up to the current timestamp
        current_data = df[df['timestamp'] <= current_time]
        for track_id in track_ids:
            track_data = current_data[current_data['track_id'] == track_id]
            x = track_data['center_x'].values
            y = track_data['center_y'].values
            lines[track_id].set_data(x, y)
        ax.set_title(f"Time: {current_time}")
        return list(lines.values())

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(timestamps),
                                  init_func=init, blit=True, repeat=False)

    # Set up labels and legend, then show the plot
    ax.legend(title="Track ID")
    plt.xlabel("Center X")
    plt.ylabel("Center Y")
    plt.title("Track Paths Animation over Time")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate track paths from a CSV file.")
    parser.add_argument('csv_path', type=str, help="Path to the CSV file")
    args = parser.parse_args()
    main(args.csv_path)