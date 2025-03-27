# -*- coding: utf-8 -*-
"""
Syringe Track Merging Script

This script reads YOLO tracking data from a CSV file, where fragmented tracks
(different track_ids for the same object over time) or untracked detections
(track_id = -1) exist. It merges these fragments into a specified number
of consistent tracks based on proximity, optional velocity prediction, and
optional volume similarity.

Designed for post-processing a complete dataset.
"""

import argparse  # To handle command-line arguments for input/output files

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# --- Configuration Parameters ---
# --- MUST BE ADJUSTED based on your specific video/data ---
NUM_SYRINGES = 2       # The exact number of syringes known to be present.
MAX_DISTANCE = 100.0   # Max distance (pixels) between a detection and a track's
                       # (predicted) position to be considered a match.
                       # CRITICAL PARAMETER - NEEDS TUNING!
MAX_LOST_FRAMES = 15   # Max consecutive timestamps a track can be missed
                       # before being considered lost/deactivated. Tune based on
                       # frequency and duration of tracking dropouts.
VOLUME_WEIGHT = 0.05   # Weight factor for volume difference in the cost function.
                       # Range: 0 (ignore volume) to ~0.5. Tune if volume is a
                       # reliable feature for distinguishing syringes.
                       # Cost = distance + VOLUME_WEIGHT * relative_vol_diff * MAX_DISTANCE
USE_VELOCITY_PREDICTION = True # Use simple linear velocity prediction to bridge gaps.
VELOCITY_DAMPING = 0.8 # Smoothing factor for velocity (0=heavy damping, 1=no damping).
                       # Used if USE_VELOCITY_PREDICTION is True.

# --- Column Names ---
# --- Adjust these if your CSV columns have different names ---
TS_COL = 'timestamp'
ID_COL = 'track_id'
X_COL = 'center_x'
Y_COL = 'center_y'
# Choose ONE representative volume column if using volume matching
# If not using volume (VOLUME_WEIGHT=0), this can be any column or None.
VOL_COL = 'volume_D0.45'
# --- End Configuration ---


# --- Helper Class for Managing Active Tracks ---
class ActiveTrack:
    """ Represents the state of a consistently tracked syringe. """
    def __init__(self, consistent_id, timestamp, x, y, volume, original_id):
        self.id = consistent_id
        self.last_timestamp = timestamp
        self.last_x = x
        self.last_y = y
        self.last_volume = volume
        # Keep track of all original YOLO track IDs that have been mapped to this consistent track
        self.original_ids = {original_id} if original_id != -1 else set()
        self.vx = 0.0  # Estimated velocity in x
        self.vy = 0.0  # Estimated velocity in y
        self.missed_frames = 0 # Counter for consecutive missed timestamps
        # Store recent history for potentially more stable velocity calculation
        self.history = [(timestamp, x, y)]
        self.is_active = True # Flag to manage active status

    def update(self, timestamp, x, y, volume, original_id):
        """ Updates the track state with a new matched detection. """
        dt = timestamp - self.last_timestamp

        # Update velocity (optional)
        if USE_VELOCITY_PREDICTION and dt > 1e-6: # Avoid division by zero
            # Calculate velocity based on the change from the last point in history
            if len(self.history) > 0:
                prev_t, prev_x, prev_y = self.history[-1]
                time_diff = timestamp - prev_t
                if time_diff > 1e-6:
                    # Calculate instantaneous velocity
                    new_vx = (x - prev_x) / time_diff
                    new_vy = (y - prev_y) / time_diff
                    # Apply exponential moving average (damping) for smoother velocity
                    self.vx = VELOCITY_DAMPING * self.vx + (1 - VELOCITY_DAMPING) * new_vx
                    self.vy = VELOCITY_DAMPING * self.vy + (1 - VELOCITY_DAMPING) * new_vy
            else: # Fallback if history is somehow empty (shouldn't happen after init)
                 self.vx = (x - self.last_x) / dt if dt > 1e-6 else 0.0
                 self.vy = (y - self.last_y) / dt if dt > 1e-6 else 0.0
        else:
            # Reset velocity if not used or dt is too small
            self.vx = 0.0
            self.vy = 0.0

        # Update core state variables
        self.last_timestamp = timestamp
        self.last_x = x
        self.last_y = y
        self.last_volume = volume
        if original_id != -1:
             self.original_ids.add(original_id)
        self.missed_frames = 0 # Reset missed counter on successful update

        # Maintain a short history (e.g., last 5 points)
        self.history.append((timestamp, x, y))
        if len(self.history) > 5:
             self.history.pop(0)

    def predict_position(self, timestamp):
        """ Predicts the track's position at a future timestamp. """
        dt = timestamp - self.last_timestamp
        if USE_VELOCITY_PREDICTION and self.is_active:
            # Simple linear prediction: pos = last_pos + velocity * time_delta
            predicted_x = self.last_x + self.vx * dt
            predicted_y = self.last_y + self.vy * dt
        else:
            # If not using velocity or track is inactive, predict it stays at last known position
            predicted_x = self.last_x
            predicted_y = self.last_y
        return predicted_x, predicted_y

    def increment_missed(self):
        """ Increments the missed frame counter. Deactivates if threshold is exceeded. """
        if self.is_active:
            self.missed_frames += 1
            if self.missed_frames > MAX_LOST_FRAMES:
                self.is_active = False # Deactivate the track
                print(f"    - Deactivating Track {self.id} (missed {self.missed_frames} frames)")
                # Optional: Reset velocity when deactivated?
                # self.vx = 0.0
                # self.vy = 0.0


# --- Main Processing Function ---
def merge_tracks(df):
    """
    Merges fragmented tracks in the DataFrame into NUM_SYRINGES consistent tracks.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame sorted by timestamp.

    Returns:
        pd.DataFrame: DataFrame with an added 'consistent_track_id' column.
                      Rows that couldn't be assigned remain NaN.
    """
    df['consistent_track_id'] = pd.NA # Use pandas NA for better type handling

    active_tracks = {} # Dictionary: {consistent_id: ActiveTrack object}
    next_consistent_id = 1

    # Get all unique timestamps present in the data, sorted
    unique_timestamps = sorted(df[TS_COL].unique())
    print(f"Processing {len(unique_timestamps)} unique timestamps...")

    for t_idx, timestamp in enumerate(unique_timestamps):

        if (t_idx + 1) % 100 == 0: # Print progress periodically
             print(f"  Processed {t_idx + 1}/{len(unique_timestamps)} timestamps...")

        # Get all rows for this specific timestamp
        current_detections_df = df[df[TS_COL] == timestamp].copy()

        # CRITICAL: Drop rows where essential coordinates are NaN *for this timestamp*
        # This handles rows that exist for the timestamp but lack detection data.
        valid_detections_df = current_detections_df.dropna(subset=[X_COL, Y_COL])

        # --- Handle Timestamps with NO Valid Detections ---
        if valid_detections_df.empty:
             # This happens if the frame originally had no detections, OR
             # if all detection rows for this timestamp had NaN coordinates.
             # We only need to update the state of currently active tracks.
             if active_tracks: # Only if there are tracks to update
                 # print(f"Timestamp {timestamp}: No valid detections. Incrementing missed count for active tracks.")
                 for track_id in list(active_tracks.keys()): # Iterate over keys copy
                     track = active_tracks[track_id]
                     track.increment_missed()
                     if not track.is_active: # Remove if deactivated by increment_missed
                          del active_tracks[track_id]
             continue # Skip to the next timestamp

        # --- Process Timestamps WITH Valid Detections ---
        detections = [
            {'index': idx, 'x': row[X_COL], 'y': row[Y_COL], 'vol': row[VOL_COL], 'orig_id': row[ID_COL]}
            for idx, row in valid_detections_df.iterrows()
        ]
        valid_detections_df.index.tolist() # Original df indices

        # Get currently active tracks
        currently_active_track_ids = [tid for tid, trk in active_tracks.items() if trk.is_active]
        num_active_tracks = len(currently_active_track_ids)
        num_detections = len(detections)

        # --- Initialization Step ---
        # If no tracks are active yet, initialize with the first valid detections
        if num_active_tracks == 0 and num_detections > 0:
            print(f"Timestamp {timestamp}: Initializing first tracks...")
            for i in range(min(num_detections, NUM_SYRINGES)):
                det = detections[i]
                new_id = next_consistent_id
                active_tracks[new_id] = ActiveTrack(new_id, timestamp, det['x'], det['y'], det['vol'], det['orig_id'])
                df.loc[det['index'], 'consistent_track_id'] = new_id
                print(f"  -> Initialized Track {new_id} with detection index {det['index']} (orig_id {det['orig_id']})")
                next_consistent_id += 1
            # Any remaining detections in this first frame are initially unassigned
            # Proceed to the next timestamp after initialization
            continue

        # --- Matching Step (when active tracks and detections exist) ---
        if num_active_tracks > 0 and num_detections > 0:
            cost_matrix = np.full((num_detections, num_active_tracks), np.inf)
            predicted_positions = {}

            # Calculate cost matrix: rows = detections, cols = active tracks
            for j, track_id in enumerate(currently_active_track_ids):
                track = active_tracks[track_id]
                pred_x, pred_y = track.predict_position(timestamp)
                predicted_positions[track_id] = (pred_x, pred_y)

                for i, det in enumerate(detections):
                    # --- Calculate Distance Cost ---
                    dist = np.sqrt((det['x'] - pred_x)**2 + (det['y'] - pred_y)**2)

                    # --- Calculate Volume Cost (Optional) ---
                    vol_cost = 0.0
                    # Check if volume data is valid and weight is positive
                    if VOLUME_WEIGHT > 0 and pd.notna(track.last_volume) and track.last_volume > 1e-6 \
                       and pd.notna(det['vol']) and det['vol'] > 1e-6:
                        # Relative difference to avoid scale issues
                        vol_diff = abs(det['vol'] - track.last_volume) / max(track.last_volume, det['vol'])
                        # Scale volume cost relative to MAX_DISTANCE threshold
                        vol_cost = vol_diff * VOLUME_WEIGHT * MAX_DISTANCE

                    # --- Total Cost ---
                    cost = dist + vol_cost

                    # Only consider matches within the distance threshold for the assignment algorithm
                    # We still record the cost even if > MAX_DISTANCE here, filtering happens after assignment.
                    cost_matrix[i, j] = cost

            # --- Assignment using Hungarian Algorithm (scipy.optimize.linear_sum_assignment) ---
            # Finds the assignment pairs that minimize the total cost.
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_detection_indices_current_step = set()
            assigned_track_ids_current_step = set()

            # print(f"Timestamp {timestamp}: Matching {num_detections} detections to {num_active_tracks} active tracks.")

            # Process optimal assignments, filtering by MAX_DISTANCE
            for i, j in zip(row_ind, col_ind):
                cost = cost_matrix[i, j]
                # IMPORTANT: Filter assignments based on the cost threshold AFTER optimal assignment
                if cost < MAX_DISTANCE:
                    detection = detections[i]
                    track_id = currently_active_track_ids[j]
                    track = active_tracks[track_id]

                    # Assign consistent ID in the DataFrame and update the track state
                    df.loc[detection['index'], 'consistent_track_id'] = track_id
                    track.update(timestamp, detection['x'], detection['y'], detection['vol'], detection['orig_id'])

                    assigned_detection_indices_current_step.add(i)
                    assigned_track_ids_current_step.add(track_id)
                    # print(f"  -> Matched Det idx {detection['index']} (orig {detection['orig_id']}) to Track {track_id} (cost {cost:.2f})")
                # else:
                    # print(f"  -> Assignment Rejected: Det idx {detections[i]['index']} to Track {currently_active_track_ids[j]} (cost {cost:.2f} >= {MAX_DISTANCE})")

        # --- Handle Unmatched Detections (Potential New Tracks) ---
        # Indices relative to the 'detections' list for this timestamp
        unmatched_detection_local_indices = set(range(num_detections)) - assigned_detection_indices_current_step

        num_active_now = len([tid for tid, trk in active_tracks.items() if trk.is_active])

        if num_active_now < NUM_SYRINGES and unmatched_detection_local_indices:
            num_new_tracks_needed = NUM_SYRINGES - num_active_now
            new_tracks_added_this_step = 0

            # Sort indices to process consistently (e.g., by original DataFrame index)
            sorted_unmatched_local_indices = sorted(list(unmatched_detection_local_indices), key=lambda i: detections[i]['index'])

            for local_idx in sorted_unmatched_local_indices:
                 if new_tracks_added_this_step < num_new_tracks_needed:
                     det = detections[local_idx]
                     new_id = next_consistent_id
                     active_tracks[new_id] = ActiveTrack(new_id, timestamp, det['x'], det['y'], det['vol'], det['orig_id'])
                     df.loc[det['index'], 'consistent_track_id'] = new_id
                     assigned_track_ids_current_step.add(new_id) # Mark as assigned
                     print(f"Timestamp {timestamp}: Initialized NEW Track {new_id} with unmatched Det idx {det['index']} (orig {det['orig_id']})")
                     next_consistent_id += 1
                     new_tracks_added_this_step += 1
                 else:
                     break # Stop if we have reached the target number of syringes

        # --- Handle Unmatched Active Tracks (Increment Missed Count, Deactivate if Needed) ---
        if num_active_tracks > 0: # Only if there were tracks to potentially miss
             missed_track_ids = set(currently_active_track_ids) - assigned_track_ids_current_step
             for track_id in list(active_tracks.keys()): # Iterate over copy of keys
                 if track_id in missed_track_ids:
                     track = active_tracks[track_id]
                     track.increment_missed()
                     if not track.is_active: # Remove if deactivated
                          del active_tracks[track_id]
                 # Reset missed counter for tracks that were *just* activated or updated
                 elif track_id in assigned_track_ids_current_step:
                      active_tracks[track_id].missed_frames = 0
                      active_tracks[track_id].is_active = True # Ensure it's marked active


        # --- Final check on active track count and potentially remove inactive ones ---
        # This loop ensures any track deactivated in earlier steps is removed if somehow missed
        for track_id in list(active_tracks.keys()):
             if not active_tracks[track_id].is_active:
                 print(f"Timestamp {timestamp}: Cleaning up deactivated track {track_id}.")
                 del active_tracks[track_id]

    print("\n--- Track Merging Complete ---")
    return df

# --- Main Execution ---
if __name__ == "__main__":
    # Set up argument parser for input and output files
    parser = argparse.ArgumentParser(description="Merge fragmented YOLO tracks from a CSV file.")
    parser.add_argument("input_csv", help="Path to the input CSV file containing tracking data.")
    parser.add_argument("output_csv", help="Path to save the output CSV file with merged tracks.")

    args = parser.parse_args()

    input_file = args.input_csv
    output_file = args.output_csv

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("\nConfiguration:")
    print(f"  NUM_SYRINGES = {NUM_SYRINGES}")
    print(f"  MAX_DISTANCE = {MAX_DISTANCE}")
    print(f"  MAX_LOST_FRAMES = {MAX_LOST_FRAMES}")
    print(f"  VOLUME_WEIGHT = {VOLUME_WEIGHT} (using column: {VOL_COL if VOLUME_WEIGHT > 0 else 'N/A'})")
    print(f"  USE_VELOCITY_PREDICTION = {USE_VELOCITY_PREDICTION}")
    print(f"  VELOCITY_DAMPING = {VELOCITY_DAMPING if USE_VELOCITY_PREDICTION else 'N/A'}\n")

    # --- Load Data ---
    try:
        # Adjust read_csv parameters if needed (e.g., separator, header row)
        df = pd.read_csv(input_file, sep=None, engine='python', na_values=['nan', 'NaN', ''])
        # sep=None lets pandas auto-detect separator, common for CSVs. Use sep=',' or sep='\t' if needed.
        print(f"Successfully loaded {len(df)} rows from {input_file}")

        # Basic check for required columns
        required_cols = [TS_COL, ID_COL, X_COL, Y_COL]
        if VOLUME_WEIGHT > 0 and VOL_COL:
             required_cols.append(VOL_COL)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             raise ValueError(f"Missing required columns in input file: {missing_cols}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        exit(1)
    except Exception as e:
        print(f"Error loading or parsing CSV file: {e}")
        exit(1)

    # --- Data Preprocessing ---
    try:
        print("\n--- Preprocessing Data ---")
        # Ensure correct data types, coercing errors to NaN
        numeric_cols = [TS_COL, ID_COL, X_COL, Y_COL]
        if VOL_COL and VOL_COL in df.columns: # Only convert VOL_COL if it exists
             numeric_cols.append(VOL_COL)

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Critical: Sort by timestamp BEFORE processing
        df = df.sort_values(by=TS_COL).reset_index(drop=True)
        print(f"Sorted by '{TS_COL}'.")

        # Handle rows where timestamp itself might be missing (shouldn't happen if sorted)
        df = df.dropna(subset=[TS_COL])

        # Fill NaN original track_ids with -1 (consistent representation for untracked)
        df[ID_COL] = df[ID_COL].fillna(-1).astype(int)
        print(f"Filled NaN '{ID_COL}' with -1.")

        # Handle NaN volumes if VOL_COL exists and volume matching is used
        if VOLUME_WEIGHT > 0 and VOL_COL and VOL_COL in df.columns:
            # Option 1: Fill with 0 (simple, assumes 0 volume if not detected)
            df[VOL_COL] = df[VOL_COL].fillna(0.0)
            # Option 2: Fill with median/mean (might be better if 0 is a valid value)
            # median_vol = df[VOL_COL].median()
            # df[VOL_COL] = df[VOL_COL].fillna(median_vol)
            print(f"Filled NaN '{VOL_COL}' with 0.0.")

        # Report rows that will be ignored due to missing essential coordinates (X or Y)
        initial_rows = len(df)
        # Keep rows even if X/Y are NaN initially, they are handled per-timestamp later
        # df_valid_coords = df.dropna(subset=[X_COL, Y_COL])
        # ignored_rows = initial_rows - len(df_valid_coords)
        # print(f"Note: {ignored_rows} rows have NaN in '{X_COL}' or '{Y_COL}' and will be ignored during matching.")
        print(f"Preprocessing complete. Starting with {len(df)} rows.")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Run Track Merging ---
    if not df.empty:
        try:
            print("\n--- Starting Track Merging Process ---")
            merged_df = merge_tracks(df.copy()) # Operate on a copy

            # --- Analyze and Save Results ---
            print("\n--- Analyzing Results ---")
            print("\nValue Counts for Consistent Track IDs (including unassigned NaN):")
            print(merged_df['consistent_track_id'].value_counts(dropna=False))

            assigned_count = merged_df['consistent_track_id'].notna().sum()
            unassigned_count = merged_df['consistent_track_id'].isna().sum()
            print(f"\nTotal Detections Processed: {len(merged_df)}")
            print(f"  Assigned to a consistent track: {assigned_count}")
            print(f"  Remained unassigned (NaN): {unassigned_count}")


            # Further analysis: Check which original IDs mapped to which consistent ID
            if assigned_count > 0:
                 print("\nMapping of Original IDs to Consistent IDs:")
                 # Filter out rows with no consistent ID assigned
                 mapping_df = merged_df.dropna(subset=['consistent_track_id'])
                 # Ensure consistent_track_id is integer for grouping
                 mapping_df['consistent_track_id'] = mapping_df['consistent_track_id'].astype(int)
                 # Group by consistent ID and list unique original IDs (excluding -1)
                 grouped_mapping = mapping_df[mapping_df[ID_COL] != -1].groupby('consistent_track_id')[ID_COL].apply(lambda x: sorted(list(set(x))))
                 if not grouped_mapping.empty:
                      print(grouped_mapping)
                 else:
                      print("  No original IDs (other than -1) were mapped to consistent tracks.")

            # --- Save Output ---
            try:
                 merged_df.to_csv(output_file, index=False, na_rep='NaN')
                 print(f"\nSuccessfully saved merged tracks to '{output_file}'")
            except Exception as e:
                 print(f"\nError saving output file: {e}")

        except Exception as e:
            print(f"\nAn error occurred during the merge_tracks function: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

    else:
        print("\nDataFrame is empty after loading/preprocessing, cannot perform merging.")
