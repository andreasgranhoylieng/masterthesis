import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

def main(csv_path: str) -> None:
    # Read the CSV into a dataframe.
    df: pd.DataFrame = pd.read_csv(csv_path)
    # Sort the dataframe by timestamp to ensure frames are in order.
    df.sort_values('timestamp', inplace=True)

    # Use every unique timestamp (even if some rows have NaN values).
    unique_timestamps = sorted(df['timestamp'].unique())
    # Get the unique track IDs.
    track_ids = df['track_id'].dropna().unique()

    # Set up the plot.
    fig, ax = plt.subplots()
    # Create a line for each track.
    lines = {track_id: ax.plot([], [], marker='o', label=f'Track {track_id}')[0] for track_id in track_ids}

    def init() -> List:
        """Initialize the plot limits and return the line objects."""
        margin: float = 10.0  # Adjust margin if needed.
        ax.set_xlim(df['center_x'].min() - margin, df['center_x'].max() + margin)
        ax.set_ylim(df['center_y'].min() - margin, df['center_y'].max() + margin)
        return list(lines.values())

    def update(frame: int) -> List:
        """
        Update the plot for the current frame.
        The cumulative valid data (non-NaN center_x and center_y) is plotted up to the current timestamp.
        """
        current_time = unique_timestamps[frame]
        # Filter cumulative data up to the current timestamp, excluding rows with NaN.
        current_data = df[(df['timestamp'] <= current_time) & (df['center_x'].notna()) & (df['center_y'].notna()) & (df['track_id'].notna())]
        for track_id in track_ids:
            track_data = current_data[current_data['track_id'] == track_id]
            x = track_data['center_x'].values
            y = track_data['center_y'].values
            lines[track_id].set_data(x, y)
        ax.set_title(f"Time: {current_time}")
        return list(lines.values())

    # Create the animation. Every unique timestamp, even empty ones, is a frame.
    ani = animation.FuncAnimation(
        fig, update, frames=len(unique_timestamps),
        init_func=init, blit=True, repeat=False
    )

    ax.legend(title="Track ID")
    plt.xlabel("Center X")
    plt.ylabel("Center Y")
    plt.title("Track Paths Animation over Time")
    plt.show()

if __name__ == '__main__':
    # Define the path to the CSV file directly.
    input_csv_path = "volume_estimation/syringe_data.csv" 
    main(input_csv_path)