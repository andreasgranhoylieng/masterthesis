# animate_tracking_history.py

import argparse
from typing import List, Dict, Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np  # For colormap
import pandas as pd

# --- Constants for Column Names ---
# Using constants makes it easier to adapt if column names change slightly
TS_COL = 'timestamp'
X_COL = 'center_x'
Y_COL = 'center_y'
# Default ID column if not specified
DEFAULT_ID_COL = 'track_id'

def animate_tracking_history(csv_path: str, id_column: str) -> None:
    """
    Reads tracking data from a CSV and animates the paths based on a specified ID column.

    Args:
        csv_path (str): Path to the input CSV file.
        id_column (str): Name of the column containing the track IDs to visualize.
    """
    # --- 1. Load Data ---
    try:
        df: pd.DataFrame = pd.read_csv(csv_path, na_values=['nan', 'NaN', ''])
        print(f"Successfully loaded {len(df)} rows from '{csv_path}'")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{csv_path}'")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- 2. Validate Columns ---
    required_cols = [TS_COL, X_COL, Y_COL, id_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # If the specified id_column is missing, maybe it's the default and the other exists?
        if id_column == DEFAULT_ID_COL and 'consistent_track_id' in df.columns:
             print(f"Warning: Default ID column '{DEFAULT_ID_COL}' not found, but 'consistent_track_id' exists.")
             print("         Consider using '--id_column consistent_track_id'")
        elif id_column != DEFAULT_ID_COL and DEFAULT_ID_COL in df.columns:
             print(f"Warning: Specified ID column '{id_column}' not found, but default '{DEFAULT_ID_COL}' exists.")
             
        print(f"Error: Missing required columns for animation: {missing_cols}")
        return

    # --- 3. Prepare Data for Animation ---
    # Ensure sorting by timestamp
    df.sort_values(TS_COL, inplace=True)
    
    # Attempt to convert ID column to numeric, coercing errors to NaN.
    # This helps handle potential non-numeric entries gracefully.
    # Use Int64 which supports NaN if possible, otherwise float.
    try:
         df[id_column] = pd.to_numeric(df[id_column], errors='coerce').astype('Int64')
    except (TypeError, ValueError):
         # If Int64 fails (e.g., float NaNs already present), fallback to float
         df[id_column] = pd.to_numeric(df[id_column], errors='coerce')

    # Get unique timestamps and track IDs (excluding NaN IDs)
    unique_timestamps = sorted(df[TS_COL].unique())
    track_ids_to_animate = sorted([tid for tid in df[id_column].dropna().unique()])

    if not track_ids_to_animate:
        print(f"Error: No valid (non-NaN) track IDs found in column '{id_column}'. Cannot animate.")
        return
    print(f"Animating tracks using column '{id_column}'. Found IDs: {track_ids_to_animate}")

    # --- 4. Set up the Plot ---
    fig, ax = plt.subplots(figsize=(12, 9)) # Adjusted size
    
    # Use a colormap for potentially many tracks
    colors = plt.cm.viridis(np.linspace(0, 1, len(track_ids_to_animate)))
    lines: Dict[Any, plt.Line2D] = {}
    for i, track_id in enumerate(track_ids_to_animate):
        # Create a line object for each track ID
        line, = ax.plot([], [], marker='o', linestyle='-', markersize=4, color=colors[i], label=f'Track {track_id}')
        lines[track_id] = line

    # --- 5. Animation Core Functions ---
    def init() -> List[plt.Line2D]:
        """Initialize plot limits and clear lines."""
        margin: float = 20.0  # Adjust margin if needed
        
        # Calculate bounds safely, handling cases with no valid data
        min_x, max_x = df[X_COL].dropna().min(), df[X_COL].dropna().max()
        min_y, max_y = df[Y_COL].dropna().min(), df[Y_COL].dropna().max()

        if pd.isna(min_x) or pd.isna(max_x) or pd.isna(min_y) or pd.isna(max_y):
             print("Warning: Could not determine plot bounds from data. Using default [0-1920, 0-1080].")
             ax.set_xlim(0, 1920) # Default fallback (e.g., common video width)
             ax.set_ylim(0, 1080) # Default fallback (e.g., common video height)
        else:
             ax.set_xlim(min_x - margin, max_x + margin)
             ax.set_ylim(min_y - margin, max_y + margin)
        
        # Invert y-axis assuming (0,0) is top-left (common in image/video)
        # Remove if your coordinate system has (0,0) at bottom-left
        ax.invert_yaxis() 

        # Reset line data
        for line in lines.values():
            line.set_data([], [])
        return list(lines.values())

    def update(frame: int) -> List[plt.Line2D]:
        """Update plot for the current frame (timestamp)."""
        current_time = unique_timestamps[frame]
        
        # Filter cumulative data up to the current timestamp
        # Include only rows with valid coordinates and a valid ID in the specified column
        current_data = df[
            (df[TS_COL] <= current_time) &
            df[X_COL].notna() &
            df[Y_COL].notna() &
            df[id_column].notna() # Use the specified ID column here
        ]
        
        # Update data for each track line
        for track_id in track_ids_to_animate:
            # Filter data specific to this track ID using the specified column
            track_data = current_data[current_data[id_column] == track_id] 
            x = track_data[X_COL].values
            y = track_data[Y_COL].values
            # Update the corresponding line object
            if track_id in lines: # Ensure line exists
                 lines[track_id].set_data(x, y)
                 
        # Update the plot title dynamically
        ax.set_title(f"Timestamp: {current_time:.3f}")
        return list(lines.values())

    # --- 6. Create and Display Animation ---
    # Create the animation object.
    # 'blit=True' can improve performance but might cause issues on some backends.
    # 'interval' is the delay between frames in milliseconds.
    ani = animation.FuncAnimation(
        fig, update, frames=len(unique_timestamps),
        init_func=init, blit=True, repeat=False, interval=50 
    )

    # Configure final plot appearance
    ax.legend(title=f"{id_column.replace('_', ' ').title()}") # Dynamic legend title
    ax.set_xlabel(X_COL)
    ax.set_ylabel(Y_COL)
    plt.title(f"{id_column.replace('_', ' ').title()} Paths Animation") # Dynamic main title
    plt.grid(True)
    plt.tight_layout()
    
    # Display the animation window
    try:
        plt.show()
    except Exception as e:
         print(f"\nError displaying animation plot: {e}")
         print("This might happen if running in an environment without a display.")
         print("Consider saving the animation to a file instead (requires ffmpeg).")
         # Example save command:
         # print("Attempting to save animation to 'output_animation.mp4'...")
         # try:
         #     ani.save('output_animation.mp4', writer='ffmpeg', fps=20, dpi=150)
         #     print("Animation saved successfully.")
         # except Exception as save_e:
         #     print(f"Error saving animation: {save_e}")


# --- 7. Main Execution Block ---
if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Animate tracking history from a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help message
    )
    
    # Positional argument for the input file
    parser.add_argument(
        "input_csv", 
        help="Path to the input CSV file containing tracking data."
    )
    
    # Optional argument for specifying the ID column
    parser.add_argument(
        "--id_column", 
        default=DEFAULT_ID_COL, 
        help=f"Name of the column containing track IDs to animate. Use 'consistent_track_id' for data processed by the merging script."
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Run the animation function with the provided arguments
    animate_tracking_history(args.input_csv, args.id_column)