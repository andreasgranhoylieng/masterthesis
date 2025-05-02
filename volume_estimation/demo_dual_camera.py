import csv
import math
import os
import time
from collections import deque, defaultdict
import sys # Added for exit
import traceback # Added for detailed error printing

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Define the Active Zone class ---
class ActiveZone:
    """Represents a named rectangular area of interest."""
    def __init__(self, name: str, rect: tuple):
        """
        Initializes an ActiveZone.

        Args:
            name (str): A unique name for the zone (e.g., "Disposal Bin", "Prep Area").
            rect (tuple): A tuple defining the zone rectangle in (x1, y1, x2, y2) format.
                          Coordinates are relative to the frame being processed.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("ActiveZone name must be a non-empty string.")
        if not (isinstance(rect, tuple) or isinstance(rect, list)) or len(rect) != 4:
            raise ValueError("ActiveZone rect must be a tuple or list of 4 numbers.")
        try:
            # Ensure coordinates are integers and valid (x1 < x2, y1 < y2)
            self.rect = tuple(map(int, rect))
            if not (self.rect[0] < self.rect[2] and self.rect[1] < self.rect[3]):
                 raise ValueError("Rectangle coordinates must be (x1, y1, x2, y2) with x1 < x2 and y1 < y2.")
        except (ValueError, TypeError) as e:
             raise ValueError(f"Invalid rectangle coordinates: {rect}. {e}")

        self.name = name

    def __repr__(self):
        return f"ActiveZone(name='{self.name}', rect={self.rect})"
# --- End Active Zone class ---


# --- Define the Syringe Volume Estimator class ---
class SyringeVolumeEstimator:
    """
    Handles detection, tracking, and volume estimation of syringes using YOLO Pose for a SINGLE camera stream.
    Also identifies which ActiveZone a syringe is in based on its configured zones.
    """
    def __init__(self, model_path: str, possible_diameters_cm: list[float], active_zones: list[ActiveZone] = None, area_threshold: float = 0.8, device_preference: str = None):
        """
        Initialize the YOLO model, device, ActiveZones, and other parameters.

        Args:
            model_path (str): Path to the trained YOLO Pose model (e.g., 'best.pt').
            possible_diameters_cm (list[float]): List of possible syringe diameters in cm
                                                 that the model should calculate volumes for.
            active_zones (list[ActiveZone], optional): A list of ActiveZone objects RELEVANT TO THIS CAMERA. Defaults to [].
            area_threshold (float, optional): The minimum proportion of the syringe's
                                              bounding box area that must be inside an AOI
                                              to consider it 'in' that zone. Defaults to 0.8 (80%).
            device_preference (str, optional): Preferred device ('cuda', 'mps', 'cpu'). If None, auto-detects.
        """
        self.model_path = model_path
        if not os.path.exists(self.model_path):
             raise FileNotFoundError(f"Model file not found: {self.model_path}. Please check the path.")

        try:
            self.model = YOLO(self.model_path).eval()
        except Exception as e:
             print(f"Error loading YOLO model from {self.model_path}: {e}")
             print("Please ensure the model path is correct, the file exists, and dependencies are installed.")
             raise
         
        
        self.device = None # <<< --- ADD THIS LINE --- Initialize self.device to None first


        # Set device based on preference or availability
        if device_preference and device_preference in ['cuda', 'mps', 'cpu']:
            self.device = device_preference
            # Add validation if needed (e.g., check torch.cuda.is_available() if 'cuda')
            if self.device == 'cuda' and not torch.cuda.is_available():
                print(f"Warning: Preferred device 'cuda' not available. Falling back.")
                self.device = None
            elif self.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                 print(f"Warning: Preferred device 'mps' not available. Falling back.")
                 self.device = None

        if self.device is None: # Auto-detect if no valid preference
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        print(f"Estimator using device: {self.device}")
        self.model.to(self.device)

        # Store possible syringe diameters
        if not isinstance(possible_diameters_cm, list) or not all(isinstance(d, (float, int)) and d > 0 for d in possible_diameters_cm):
             raise ValueError("possible_diameters_cm must be a list of positive numbers.")
        self.possible_diameters = sorted(possible_diameters_cm)
        # print(f"Possible syringe diameters (cm): {self.possible_diameters}") # Printed in main now

        # --- Active Zone Configuration ---
        self.active_zones = active_zones if active_zones is not None else []
        if not isinstance(self.active_zones, list) or not all(isinstance(zone, ActiveZone) for zone in self.active_zones):
             raise TypeError("active_zones must be a list of ActiveZone objects.")
        zone_names = [zone.name for zone in self.active_zones]
        if len(zone_names) != len(set(zone_names)):
            # Allowing duplicate names might cause ambiguity in zone reporting. Enforcing uniqueness within this estimator's zones.
            raise ValueError(f"ActiveZone names for this estimator must be unique: {zone_names}")
        self.area_threshold = max(0.1, min(1.0, area_threshold)) # Clamp threshold between 0.1 and 1.0
        print(f"  Zones configured for this estimator: {[zone.name for zone in self.active_zones]}")
        # print(f"  Zone Area Threshold: {self.area_threshold * 100:.0f}%") # Printed in main now

        # Internal state for FPS calculation
        self.last_timestamps = deque(maxlen=10) # Use more samples for smoother FPS
        self.save_video = False # Controlled externally by the run script


    def draw_volume_table(self, frame: np.ndarray, volumes: list[tuple[float, float]], table_x: int, table_y: int, track_id: int) -> None:
        """Draw a table on the frame showing diameters and volumes with track ID."""
        frame_h, frame_w = frame.shape[:2]
        table_width = 250
        row_height = 25
        header_height = 50
        table_height = header_height + len(volumes) * row_height + 5 # Extra padding

        # Adjust position if table goes off-screen
        if table_x + table_width > frame_w - 10:
            table_x = frame_w - table_width - 10 # Move left
        if table_y + table_height > frame_h - 10:
            table_y = frame_h - table_height - 10 # Move up
        if table_x < 10: table_x = 10
        if table_y < 10: table_y = 10

        # Ensure integer coordinates for drawing
        table_x, table_y = int(table_x), int(table_y)

        try:
            # Create a transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (table_x, table_y), (table_x + table_width, table_y + table_height), (210, 210, 210), -1) # Light grey bg
            alpha = 0.6 # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        except Exception as e:
            print(f"Warning: Error drawing table background: {e}") # Catch potential drawing errors


        # --- Draw text content ---
        text_color = (0, 0, 0) # Black
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Draw Track ID
        cv2.putText(frame, f"Syringe ID: {track_id}", (table_x + 10, table_y + 20),
                    font, 0.6, text_color, thickness + 1) # Slightly larger ID

        # Draw Headers
        cv2.putText(frame, "Diam (cm)", (table_x + 10, table_y + header_height - 10),
                    font, font_scale, text_color, thickness)
        cv2.putText(frame, "Volume (mL)", (table_x + 120, table_y + header_height - 10),
                    font, font_scale, text_color, thickness)

        # Draw Volume Rows
        for i, (diameter, volume) in enumerate(volumes):
            row_y = table_y + header_height + i * row_height + (row_height // 2) # Center text in row
            cv2.putText(frame, f"{diameter:.2f}", (table_x + 10, row_y),
                        font, font_scale, text_color, thickness)

            if volume is not None and not math.isnan(volume) and volume >= 0:
                # Display valid, non-negative volumes
                display_vol = f"{volume:.2f}"
            else:
                # Display N/A for invalid, NaN, or negative volumes
                display_vol = "N/A"

            cv2.putText(frame, display_vol, (table_x + 120, row_y),
                        font, font_scale, text_color, thickness)


    def draw_fps_counter(self, frame: np.ndarray, prefix: str = "FPS") -> None:
        """Draw a counter on the frame showing the average FPS."""
        current_timestamp = time.monotonic() # Use monotonic clock for FPS
        self.last_timestamps.append(current_timestamp)

        if len(self.last_timestamps) > 1:
            try:
                time_diffs = np.diff(list(self.last_timestamps))
                # Ignore large gaps that might occur if processing stalls
                valid_diffs = time_diffs[time_diffs < 1.0] # Only consider frame times < 1 sec
                if len(valid_diffs) > 0:
                    avg_time = np.mean(valid_diffs)
                    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
                else:
                     avg_fps = 0 # Avoid division by zero if no valid diffs

                text = f"{prefix}: {avg_fps:.1f}" # Add prefix (e.g., "Manikin FPS")
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Position at top-left
                rect_x1, rect_y1 = 8, 10
                rect_x2, rect_y2 = rect_x1 + text_width + 12, rect_y1 + text_height + baseline + 5

                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1) # White background
                cv2.putText(frame, text, (rect_x1 + 6, rect_y2 - baseline), font, font_scale, (0, 0, 0), thickness) # Black text
            except Exception as e:
                print(f"Warning: Error calculating FPS: {e}") # Avoid crashing


    def process_frame(self, frame: np.ndarray, timestamp: float, writer: csv.writer = None, camera_source: str = "unknown") -> tuple[np.ndarray, list]:
        """
        Process a single frame: detect, track, check zones, estimate volumes, draw, and log.

        Args:
            frame (np.ndarray): The input video frame.
            timestamp (float): The timestamp associated with the frame (e.g., seconds).
            writer (csv.writer, optional): CSV writer object to log raw data. Defaults to None.
            camera_source (str, optional): Identifier for the camera source (e.g., 'manikin', 'syringes'). Defaults to "unknown".


        Returns:
            tuple:
                - annotated_frame (np.ndarray): Frame with drawings (zones, detections, tables, FPS).
                - detections (list[dict]): List of detected syringes with their info:
                  Each dict contains: 'id', 'zone', 'volumes', 'bbox', 'center', 'in_active_zone_flag'.
                  Returns an empty list if no valid syringes are detected or tracking fails.
        """
        original_frame_height, original_frame_width = frame.shape[:2]
        annotated_frame = frame.copy() # Work on a copy for drawing

        # --- Draw Active Zones relevant to THIS estimator/camera ---
        for zone in self.active_zones:
            x1, y1, x2, y2 = zone.rect
            # Clamp coordinates to be within frame boundaries for drawing
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_frame_width - 1, x2), min(original_frame_height - 1, y2)
            if x1 >= x2 or y1 >= y2: continue # Skip drawing invalid rects

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta outline
            # Position zone label inside or outside the box based on position
            label_y = y1 - 10 if y1 > 30 else y1 + 20
            label_y = max(15, min(label_y, original_frame_height - 5)) # Keep label in frame
            cv2.putText(annotated_frame, zone.name, (x1 + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # --- Draw FPS Counter ---
        # Include camera source in FPS label for clarity
        fps_prefix = f"{camera_source.capitalize()} FPS"
        self.draw_fps_counter(annotated_frame, prefix=fps_prefix)

        # --- YOLO Detection and Tracking ---
        processed_detections = [] # Initialize empty list for results
        try:
            # Use persist=True for tracking across frames
            results = self.model.track(source=frame, # Use original frame
                                       persist=True,
                                       tracker='bytetrack.yaml', # Or 'botsort.yaml'
                                       verbose=False,          # Suppress console output from YOLO
                                       conf=0.5,               # Confidence threshold (adjust as needed)
                                       device=self.device,
                                       classes=[0],            # Assuming class 0 is 'syringe'
                                      )
            if not results or len(results) == 0:
                 # No results returned from tracker
                 return annotated_frame, []

            # Process results (move to CPU for numpy/cv2 operations)
            results_data = results[0].cpu() # Process the first (and likely only) result object

            # Plot default YOLO annotations (boxes, keypoints, labels) onto the frame
            # Do this *before* our custom green boxes and tables
            annotated_frame = results_data.plot(img=annotated_frame, line_width=1, font_size=0.4)

        except Exception as e:
            print(f"Error during YOLO tracking/plotting ({camera_source}): {e}")
            # Log an error row if writer is available
            if writer:
                 # Include camera_source in the error log row
                 row = [timestamp, camera_source, np.nan, np.nan, np.nan] + [np.nan for _ in self.possible_diameters] + ['TrackingError', 0]
                 writer.writerow(row)
            return annotated_frame, [] # Return frame with zones/fps, but no detections


        # --- Process each detected/tracked object ---
        # Check if tracker returned boxes and IDs
        if results_data.boxes is None or results_data.boxes.id is None:
             # No tracks found in this frame
             return annotated_frame, []

        # Iterate through detected boxes and their track IDs
        for i, box in enumerate(results_data.boxes):
            track_id = int(box.id[0]) # Get track ID for THIS camera stream

            # --- Bounding Box and Center ---
            box_coords = box.xyxy[0].numpy() # Get box coordinates (x1, y1, x2, y2)
            x1_syr, y1_syr, x2_syr, y2_syr = map(int, box_coords)
            if x1_syr >= x2_syr or y1_syr >= y2_syr: continue # Skip invalid boxes
            syringe_bbox = (x1_syr, y1_syr, x2_syr, y2_syr)
            center_x = float((x1_syr + x2_syr) / 2)
            center_y = float((y1_syr + y2_syr) / 2)
            syringe_box_area = float((x2_syr - x1_syr) * (y2_syr - y1_syr))

            # --- Zone Detection (using THIS estimator's active_zones) ---
            in_active_zone_flag = False
            detected_zone_name = "Outside" # Default
            if syringe_box_area > 0:
                best_overlap_zone = None
                max_overlap_ratio = 0.0
                for zone in self.active_zones: # Check against zones configured for this camera
                    x1_zone, y1_zone, x2_zone, y2_zone = zone.rect
                    # Calculate intersection area
                    inter_x1 = max(syringe_bbox[0], x1_zone)
                    inter_y1 = max(syringe_bbox[1], y1_zone)
                    inter_x2 = min(syringe_bbox[2], x2_zone)
                    inter_y2 = min(syringe_bbox[3], y2_zone)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    intersection_area = float(inter_w * inter_h)
                    overlap_ratio = intersection_area / syringe_box_area

                    # Find zone with highest overlap ratio above threshold
                    if overlap_ratio >= self.area_threshold and overlap_ratio > max_overlap_ratio:
                          max_overlap_ratio = overlap_ratio
                          best_overlap_zone = zone.name

                if best_overlap_zone:
                    in_active_zone_flag = True
                    detected_zone_name = best_overlap_zone
            # --- End Zone Detection ---


            # --- Volume Estimation using Keypoints ---
            volumes_for_diameters = {D: float('nan') for D in self.possible_diameters} # Initialize volumes as NaN

            # Check if keypoints are available and valid for this detection
            if results_data.keypoints is not None and len(results_data.keypoints.xy) > i and results_data.keypoints.xy[i].shape[0] >= 4:
                try:
                    kpts = results_data.keypoints.xy[i][:4].numpy() # Get first 4 keypoints
                    ll_point, ul_point, ur_point, lr_point = kpts

                    width_px = (np.linalg.norm(lr_point - ll_point) + np.linalg.norm(ur_point - ul_point)) / 2.0
                    height_px = (np.linalg.norm(ul_point - ll_point) + np.linalg.norm(ur_point - lr_point)) / 2.0

                    if width_px > 2 and height_px > 2: # Basic sanity check
                        for D_cm in self.possible_diameters:
                            scale_factor_cm_per_px = D_cm / width_px
                            H_cm = height_px * scale_factor_cm_per_px
                            if 0 < H_cm <= 25.0:
                                radius_cm = D_cm / 2.0
                                volume_ml = math.pi * (radius_cm ** 2) * H_cm
                                volumes_for_diameters[D_cm] = volume_ml

                except Exception as e:
                    # print(f"Debug: Volume calculation error for ID {track_id} ({camera_source}): {e}") # Optional debug print
                    pass # Keep volumes as NaN

            # --- Store Results for this Syringe ---
            # NOTE: camera_source is added in the main loop after this function returns
            detection_info = {
                'id': track_id,                   # ID is specific to this camera's tracker
                'zone': detected_zone_name,       # Zone name (implies camera if names are unique overall)
                'volumes': volumes_for_diameters, # Dict {diameter: volume_ml}
                'bbox': syringe_bbox,             # Tuple (x1, y1, x2, y2)
                'center': (center_x, center_y),   # Tuple (cx, cy)
                'in_active_zone_flag': in_active_zone_flag # Boolean (based on this camera's zones)
            }
            processed_detections.append(detection_info)

            # --- Log Raw Data to CSV (if writer provided) ---
            if writer:
                volumes_list = [volumes_for_diameters.get(D, np.nan) for D in self.possible_diameters]
                # Include camera_source in the CSV row
                log_row = [timestamp, camera_source, track_id, center_x, center_y] + volumes_list + \
                          [detected_zone_name, 1 if in_active_zone_flag else 0]
                try:
                    writer.writerow(log_row)
                except Exception as e:
                    print(f"Error writing row to CSV ({camera_source}): {e}")


            # --- Draw Volume Table ---
            volumes_for_table = list(volumes_for_diameters.items())
            table_x = x2_syr + 10 # Position right of the bounding box
            table_y = y1_syr      # Align with top of the bounding box
            self.draw_volume_table(annotated_frame, volumes_for_table, table_x, table_y, track_id)


        # --- Post-processing Drawing ---
        # Redraw boxes in green for syringes detected inside *any* of THIS CAMERA's active zones
        for det in processed_detections:
             if det['in_active_zone_flag']:
                 x1, y1, x2, y2 = map(int, det['bbox'])
                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green outline, thickness 2

        # Draw overall "ACTIVE ZONE DETECTED" banner IF a syringe is in one of THIS camera's zones
        if any(d['in_active_zone_flag'] for d in processed_detections):
            text = f"ACTIVE ZONE DETECTED ({camera_source})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
             # Position at bottom center
            text_x = (annotated_frame.shape[1] - text_width) // 2
            text_y = annotated_frame.shape[0] - 15
            rect_y1 = text_y - text_height - baseline - 5
            rect_y2 = annotated_frame.shape[0] - 10
            cv2.rectangle(annotated_frame, (text_x - 10, rect_y1), (text_x + text_width + 10, rect_y2), (255, 255, 255), -1) # White bg
            cv2.putText(annotated_frame, text, (text_x, text_y - baseline // 2), font, font_scale, (0, 180, 0), thickness) # Dark green text


        return annotated_frame, processed_detections


    # setup_capture is removed - capture setup done in main script
    # def setup_capture(...)


    def setup_video_writer(self, output_path: str, frame_width: int, frame_height: int, fps: float) -> cv2.VideoWriter:
        """Sets up the video writer object if save_video is True."""
        # Now takes dimensions and fps directly, as cap object might not be available here
        if not self.save_video:
            return None

        if fps <= 0: # Handle case where FPS is not available or zero
             fps = 25.0 # Use a common default FPS
             print(f"Warning: Invalid FPS provided ({fps}). Using default: {fps:.1f} FPS")

        if frame_width <= 0 or frame_height <= 0:
             print(f"Error: Invalid frame dimensions provided ({frame_width}x{frame_height}). Cannot create video writer.")
             self.save_video = False # Disable saving
             return None

        # Choose a common codec (mp4v for .mp4, XVID for .avi)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True) # Handle case where path is just filename

        try:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                 raise IOError(f"VideoWriter failed to open for path: {output_path}")
            print(f"💾 Saving processed video to: {output_path} ({frame_width}x{frame_height} @ {fps:.1f} FPS)")
            return out
        except Exception as e:
            print(f"Error initializing VideoWriter for {output_path}: {e}. Video saving disabled for this stream.")
            self.save_video = False
            return None
# --- End of SyringeVolumeEstimator Class ---


# --- Define the Syringe Test Workflow Class ---
class SyringeTestWorkflow:
    """Manages and validates a multi-step syringe handling test using input from two cameras."""

    # Define states
    STATE_IDLE = "IDLE"                     # Waiting for the correct syringe to be picked from table zone
    STATE_SYRINGE_PICKED = "SYRINGE_PICKED" # Correct syringe believed to be handled (off table), waiting for insertion
    STATE_SYRINGE_INSERTED = "SYRINGE_INSERTED" # Syringe detected in a target zone (manikin cam)

    # Define error types
    ERROR_WRONG_SYRINGE = "Wrong Syringe Picked" # Currently based on starting zone only
    ERROR_WRONG_VOLUME = "Incorrect Volume"
    ERROR_WRONG_TARGET = "Wrong Target Zone"
    ERROR_MULTI_ACTIVE = "Multiple Syringes Active" # Multiple syringes leaving table
    ERROR_LOST_TRACK = "Lost Track of Syringe"
    ERROR_PREMATURE_RETURN = "Syringe Returned Prematurely"
    ERROR_VOLUME_UNDETERMINED = "Volume Undetermined"
    ERROR_UNEXPECTED_INSERT = "Unexpected Syringe Insertion" # Inserted when not in PICKED state

    def __init__(self,
                 table_zone_names: list[str], # Zones monitored by syringe camera
                 target_zone_names: list[str], # Zones monitored by manikin camera
                 correct_starting_zone: str,
                 correct_syringe_diameter: float,
                 possible_diameters: list[float],
                 target_volume_ml: float,
                 volume_tolerance_ml: float,
                 correct_target_zone: str,
                 log_file_path: str = "workflow_log.txt"):
        """
        Initializes the workflow manager for a two-camera setup.

        Args:
            table_zone_names (list[str]): Names of zones considered 'on the table' (monitored by syringe cam).
            target_zone_names (list[str]): Names of zones considered insertion targets (monitored by manikin cam).
            correct_starting_zone (str): Name of the table zone the correct syringe should start in.
            correct_syringe_diameter (float): Diameter (cm) of the expected syringe.
            possible_diameters (list[float]): List of all diameters the estimator can check.
            target_volume_ml (float): Expected volume in the syringe upon insertion.
            volume_tolerance_ml (float): Allowed +/- tolerance for the volume.
            correct_target_zone (str): Name of the target zone the syringe should be inserted into.
            log_file_path (str): Path to the workflow log file.
        """
        self.table_zones = set(table_zone_names)
        self.target_zones = set(target_zone_names)
        # Assuming zone names are unique across both cameras now
        self.all_defined_zones = self.table_zones.union(self.target_zones)

        # --- Validate Configuration ---
        if not table_zone_names or not target_zone_names:
             raise ValueError("table_zone_names and target_zone_names cannot be empty.")
        if correct_starting_zone not in self.table_zones:
            raise ValueError(f"Config Error: Correct starting zone '{correct_starting_zone}' is not defined in table zones: {table_zone_names}")
        if correct_target_zone not in self.target_zones:
             raise ValueError(f"Config Error: Correct target zone '{correct_target_zone}' is not defined in target zones: {target_zone_names}")
        if correct_syringe_diameter not in possible_diameters:
             raise ValueError(f"Config Error: Correct syringe diameter {correct_syringe_diameter}cm is not in estimator's possible diameters: {possible_diameters}")
        if target_volume_ml <= 0 or volume_tolerance_ml < 0:
             raise ValueError("target_volume_ml must be positive, and volume_tolerance_ml must be non-negative.")

        self.correct_starting_zone = correct_starting_zone
        self.correct_diameter = correct_syringe_diameter
        self.target_volume = target_volume_ml
        self.volume_tolerance = volume_tolerance_ml
        self.correct_target_zone = correct_target_zone

        # --- State Variables ---
        self.current_state = self.STATE_IDLE
        self.active_syringe_id = None            # Track ID of the syringe being handled
                                                 # This ID's source camera depends on the state!
        self.active_id_source = None             # 'syringes' or 'manikin' - indicates which camera's ID is currently active
        self.active_syringe_start_zone = None    # Zone the active syringe originated from (table zone)
        self.active_syringe_current_zone = None  # Most recent zone of the active syringe (can be target or table)
        self.active_syringe_volume = None        # Measured volume at insertion time (from manikin cam)
        self.error_flags_this_cycle = set()      # Store errors encountered in the current pickup-insert-return cycle
        self.last_state_update_time = time.monotonic() # Track time for timeouts

        # --- Tracking History (Consolidated from both cameras) ---
        # Stores the last known zone for any ID seen by EITHER camera.
        # Zone name implicitly tells us which camera likely saw it last.
        self.syringe_last_known_zone = defaultdict(lambda: "Unknown") # {track_id: zone_name} - ID can be from either camera
        self.syringe_last_seen_time = defaultdict(float)             # {track_id: timestamp} - ID can be from either camera
        self.SYRINGE_PURGE_TIMEOUT = 15.0 # Seconds after which info about a missing syringe is purged
        self.PICKUP_TO_INSERT_TIMEOUT = 10.0 # Max time between pickup and insertion appearance
        self.INSERT_TO_RETURN_TIMEOUT = 20.0 # Max time between insertion and return appearance

        # --- Logging Setup ---
        self.log_file_path = log_file_path
        self._log_entries = [] # Store logs in memory during run
        self._clear_log_file() # Clear previous log file on initialization
        self._log_header()     # Log the configuration details

    def _clear_log_file(self):
        """Clears the content of the log file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file_path) or '.', exist_ok=True)
            with open(self.log_file_path, 'w') as f:
                f.write("") # Overwrite with empty content
        except IOError as e:
            print(f"Warning: Could not clear log file {self.log_file_path}: {e}")

    def _log(self, message: str, level: str = "INFO"):
        """Adds a timestamped message to the log (memory and file)."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} [{level:<5}] [{self.current_state}] {message}"
        print(log_message) # Print to console for real-time feedback
        self._log_entries.append(log_message)
        # Append to file immediately
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(log_message + "\n")
        except IOError as e:
            print(f"Warning: Could not write to log file {self.log_file_path}: {e}")

    def _log_error(self, error_type: str, message: str):
        """Logs an error and flags it for the current cycle."""
        self._log(f"{error_type}: {message}", level="ERROR")
        self.error_flags_this_cycle.add(error_type)

    def _log_header(self):
        """Logs the configuration parameters at the start."""
        self._log("--- Workflow Initialized (Two-Camera Mode) ---", level="SETUP")
        self._log(f"Table Zones (Syringe Cam): {list(self.table_zones)}", level="SETUP")
        self._log(f"Target Zones (Manikin Cam): {list(self.target_zones)}", level="SETUP")
        self._log(f"Correct Start Zone: '{self.correct_starting_zone}'", level="SETUP")
        self._log(f"Correct Diameter: {self.correct_diameter} cm", level="SETUP")
        self._log(f"Target Volume: {self.target_volume:.2f} +/- {self.volume_tolerance:.2f} mL", level="SETUP")
        self._log(f"Correct Target Zone: '{self.correct_target_zone}'", level="SETUP")
        self._log("--- Waiting for Action ---", level="SETUP")


    def _reset_state(self, reason: str):
        """Resets the workflow state to IDLE and clears active syringe info."""
        if self.current_state != self.STATE_IDLE: # Only log reset if not already idle
            # Log cycle summary before resetting
            if self.active_syringe_id is not None:
                 # Report ID based on the state we are coming FROM
                 id_to_report = self.active_syringe_id
                 source_cam = self.active_id_source
                 summary = f"Cycle ended for syringe ID {id_to_report} (from {source_cam} cam)."
                 if self.error_flags_this_cycle:
                     summary += f" Errors: {', '.join(sorted(list(self.error_flags_this_cycle)))}."
                 else:
                     # Check if insertion happened before declaring 'OK'
                     if self.current_state == self.STATE_SYRINGE_INSERTED:
                          summary += " Status: OK (Returned to table)."
                     else:
                          summary += " Status: Aborted before completion."

                 self._log(summary, level="SUMMARY")

            self._log(f"Resetting state to IDLE. Reason: {reason}")

        self.current_state = self.STATE_IDLE
        self.active_syringe_id = None
        self.active_id_source = None
        self.active_syringe_start_zone = None
        self.active_syringe_current_zone = None
        self.active_syringe_volume = None
        self.error_flags_this_cycle = set() # Clear errors for the new cycle
        self.last_state_update_time = time.monotonic() # Reset timer


    def _purge_old_tracks(self, current_time: float):
         """Removes information about tracks not seen by EITHER camera for a while."""
         purged_ids = [
             track_id for track_id, last_seen in self.syringe_last_seen_time.items()
             if current_time - last_seen > self.SYRINGE_PURGE_TIMEOUT
         ]
         for track_id in purged_ids:
             # Make sure we don't purge the currently active syringe if it's temporarily lost
             if track_id == self.active_syringe_id: continue

             if track_id in self.syringe_last_known_zone:
                 del self.syringe_last_known_zone[track_id]
             if track_id in self.syringe_last_seen_time:
                 del self.syringe_last_seen_time[track_id]
             # Optional: Log purge event if needed for debugging
             # self._log(f"Purged tracking info for inactive ID {track_id}", level="DEBUG")


    def update_state(self, all_detections: list[dict], current_time: float):
        """
        Updates the workflow state based on the latest detections from BOTH cameras.

        Args:
            all_detections (list[dict]): Combined list of detection dictionaries from both cameras.
                                         Each dict MUST have a 'camera_source' key ('manikin' or 'syringes').
            current_time (float): Current timestamp (e.g., from time.monotonic()).
        """
        # --- Update Tracking History (Consolidated) ---
        detections_by_source = {'manikin': [], 'syringes': []}
        current_detections_map = {} # Map (track_id, camera_source) -> detection for quick lookup

        for det in all_detections:
            if 'camera_source' not in det:
                self._log("Internal Error: Detection missing 'camera_source' key. Skipping.", level="ERROR")
                continue
            if det['id'] == -1 or det['id'] is None: continue # Skip invalid IDs

            source = det['camera_source']
            track_id = det['id']
            zone = det['zone']

            detections_by_source[source].append(det)
            current_detections_map[(track_id, source)] = det

            # Update last seen time and zone for this specific ID on this specific camera
            # Note: self.syringe_last_known_zone now stores the zone name, implying the camera
            self.syringe_last_known_zone[(track_id, source)] = zone
            self.syringe_last_seen_time[(track_id, source)] = current_time

        # --- Purge Old Tracks (using the combined history) ---
        # We need to adjust purging logic slightly or keep separate last_seen maps?
        # Let's refine purge logic: Purge an ID if it hasn't been seen on *either* camera for timeout.
        # Keep a simpler global last seen time for purging:
        global_last_seen = defaultdict(float)
        for (track_id, source), timestamp in self.syringe_last_seen_time.items():
            global_last_seen[track_id] = max(global_last_seen[track_id], timestamp)

        purged_ids = [
            track_id for track_id, last_seen in global_last_seen.items()
            if current_time - last_seen > self.SYRINGE_PURGE_TIMEOUT and track_id != self.active_syringe_id
        ]
        for track_id in purged_ids:
             # Remove from all tracking dicts
             keys_to_remove = [(tid, src) for (tid, src) in self.syringe_last_known_zone if tid == track_id]
             for key in keys_to_remove: del self.syringe_last_known_zone[key]
             keys_to_remove = [(tid, src) for (tid, src) in self.syringe_last_seen_time if tid == track_id]
             for key in keys_to_remove: del self.syringe_last_seen_time[key]
             # self._log(f"Purged tracking info for inactive ID {track_id}", level="DEBUG")


        # --- State Machine Logic ---

        # == STATE: IDLE ==
        if self.current_state == self.STATE_IDLE:
            syringes_leaving_table = []
            # Check syringe cam for syringes that just moved from a table zone to 'Outside'
            for det in detections_by_source['syringes']:
                track_id = det['id']
                current_zone = det['zone']
                last_zone = self.syringe_last_known_zone.get((track_id, 'syringes'), "Unknown")

                # Condition: Currently 'Outside' AND previously was in a table zone
                if current_zone not in self.table_zones and last_zone in self.table_zones:
                    syringes_leaving_table.append(det)
                    det['origin_zone'] = last_zone # Store where it came from

            if len(syringes_leaving_table) == 1:
                pickup_det = syringes_leaving_table[0]
                pickup_id = pickup_det['id']
                origin_zone = pickup_det['origin_zone']

                self._log(f"Syringe ID {pickup_id} (syringes cam) picked up from '{origin_zone}'. State -> {self.STATE_SYRINGE_PICKED}.")

                # Set active syringe details
                self.active_syringe_id = pickup_id
                self.active_id_source = 'syringes'
                self.active_syringe_start_zone = origin_zone
                self.active_syringe_current_zone = pickup_det['zone'] # Likely 'Outside'
                self.current_state = self.STATE_SYRINGE_PICKED
                self.error_flags_this_cycle = set() # Start fresh error set
                self.last_state_update_time = current_time # Reset timer for PICKED state

                # Check if picked from the wrong starting zone
                if origin_zone != self.correct_starting_zone:
                    self._log_error(self.ERROR_WRONG_SYRINGE, f"Picked from '{origin_zone}', expected '{self.correct_starting_zone}'.")

            elif len(syringes_leaving_table) > 1:
                ids = [d['id'] for d in syringes_leaving_table]
                self._log_error(self.ERROR_MULTI_ACTIVE, f"Multiple syringes ({ids}) picked up simultaneously from table zones.")
                # Don't change state, wait for only one to be active

            # Also check if a syringe appeared directly in a target zone (unexpected)
            for det in detections_by_source['manikin']:
                 if det['zone'] in self.target_zones:
                      last_zone = self.syringe_last_known_zone.get((det['id'], 'manikin'), "Unknown")
                      if last_zone not in self.target_zones: # Check if it just entered
                          self._log_error(self.ERROR_UNEXPECTED_INSERT, f"Syringe ID {det['id']} (manikin cam) appeared in target zone '{det['zone']}' while IDLE.")


        # == STATE: SYRINGE_PICKED ==
        elif self.current_state == self.STATE_SYRINGE_PICKED:
            # We are tracking self.active_syringe_id (which is from 'syringes' cam)
            active_syringe_pickup_id = self.active_syringe_id
            syringes_cam_det = current_detections_map.get((active_syringe_pickup_id, 'syringes'))

            # 1. Check if returned to table prematurely
            if syringes_cam_det and syringes_cam_det['zone'] in self.table_zones:
                returned_zone = syringes_cam_det['zone']
                self._log_error(self.ERROR_PREMATURE_RETURN, f"Syringe ID {active_syringe_pickup_id} returned to table zone '{returned_zone}' before insertion.")
                self._reset_state(f"Syringe returned to '{returned_zone}' prematurely")
                return # End processing for this frame

            # 2. Check if appeared in a target zone on manikin cam
            syringe_inserted = None
            for det in detections_by_source['manikin']:
                if det['zone'] in self.target_zones:
                     # Check if it JUST entered the target zone this frame
                     last_zone_manikin = self.syringe_last_known_zone.get((det['id'], 'manikin'), "Unknown")
                     if last_zone_manikin not in self.target_zones:
                           syringe_inserted = det
                           break # Assume the first one seen entering is the one

            if syringe_inserted:
                insertion_id = syringe_inserted['id']
                insertion_zone = syringe_inserted['zone']
                self._log(f"Syringe detected in target zone '{insertion_zone}' (Manikin Cam ID: {insertion_id}). Assuming pickup ID {active_syringe_pickup_id} was inserted. State -> {self.STATE_SYRINGE_INSERTED}.")

                # *** Transition: Update active ID to the one seen by manikin cam ***
                self.active_syringe_id = insertion_id
                self.active_id_source = 'manikin'
                self.active_syringe_current_zone = insertion_zone
                self.current_state = self.STATE_SYRINGE_INSERTED
                self.last_state_update_time = current_time # Reset timer for INSERTED state

                # --- Perform Insertion Checks (using manikin cam detection) ---
                # a. Check Volume: Retrieve volume for the CORRECT diameter
                volume = syringe_inserted['volumes'].get(self.correct_diameter, float('nan'))
                if volume is not None and not math.isnan(volume) and volume >= 0:
                    self.active_syringe_volume = volume
                    self._log(f"Volume measured at insertion: {self.active_syringe_volume:.2f} mL (for {self.correct_diameter}cm diameter).")
                    # Check if volume is within tolerance
                    if not (self.target_volume - self.volume_tolerance <= self.active_syringe_volume <= self.target_volume + self.volume_tolerance):
                        self._log_error(self.ERROR_WRONG_VOLUME, f"Volume {self.active_syringe_volume:.2f}mL is outside target range {self.target_volume:.2f} +/- {self.volume_tolerance:.2f}mL.")
                    else:
                         self._log("Volume is within target range.")
                else:
                    # Volume could not be determined or was invalid
                    self.active_syringe_volume = None
                    self._log_error(self.ERROR_VOLUME_UNDETERMINED, f"Could not determine valid volume for diameter {self.correct_diameter}cm at insertion.")

                # b. Check Target Zone
                if insertion_zone != self.correct_target_zone:
                    self._log_error(self.ERROR_WRONG_TARGET, f"Inserted into WRONG zone '{insertion_zone}'. Expected '{self.correct_target_zone}'.")
                else:
                     self._log(f"Inserted into CORRECT target zone '{insertion_zone}'.")

                # c. Re-check starting zone error (logged during pickup)
                if self.ERROR_WRONG_SYRINGE in self.error_flags_this_cycle:
                    self._log("Reminder: Syringe was potentially picked from the wrong starting zone.", level="WARN")

                return # End processing for this frame after insertion detected


            # 3. Check for Timeout: Picked up but not inserted
            if current_time - self.last_state_update_time > self.PICKUP_TO_INSERT_TIMEOUT:
                 self._log_error(self.ERROR_LOST_TRACK, f"Syringe ID {active_syringe_pickup_id} picked up but not seen in target zone within {self.PICKUP_TO_INSERT_TIMEOUT:.1f}s.")
                 self._reset_state("Timeout waiting for insertion")
                 return

            # 4. Check if original pickup ID disappeared from syringe cam view (and not inserted yet)
            if not syringes_cam_det:
                # Optional: Log disappearance from syringe cam if needed for debugging
                # self._log(f"Syringe ID {active_syringe_pickup_id} no longer seen by syringe cam, waiting for insertion.", level="DEBUG")
                pass


        # == STATE: SYRINGE_INSERTED ==
        elif self.current_state == self.STATE_SYRINGE_INSERTED:
            # We are tracking self.active_syringe_id (which is from 'manikin' cam)
            active_syringe_manikin_id = self.active_syringe_id
            manikin_cam_det = current_detections_map.get((active_syringe_manikin_id, 'manikin'))

            # 1. Check if returned to a table zone on syringe cam
            syringe_returned = None
            for det in detections_by_source['syringes']:
                 if det['zone'] in self.table_zones:
                      # Check if it JUST entered the table zone this frame
                      last_zone_syringes = self.syringe_last_known_zone.get((det['id'], 'syringes'), "Unknown")
                      if last_zone_syringes not in self.table_zones:
                          syringe_returned = det
                          break # Assume first syringe seen entering table is the one

            if syringe_returned:
                returned_zone = syringe_returned['zone']
                returned_id = syringe_returned['id']
                self._log(f"Syringe ID {returned_id} (syringes cam) detected entering table zone '{returned_zone}'. Assuming return of inserted syringe ID {active_syringe_manikin_id}.")
                self._reset_state(f"Workflow cycle completed, syringe returned to '{returned_zone}'")
                return # End processing

            # 2. Check if still in manikin view or moved within manikin zones
            if manikin_cam_det:
                 current_zone = manikin_cam_det['zone']
                 # Update current zone if it changed (e.g., moved outside target but still visible)
                 if current_zone != self.active_syringe_current_zone:
                      if current_zone in self.target_zones:
                          # Moved to a different target zone?
                           self._log_error(self.ERROR_WRONG_TARGET, f"Syringe moved from '{self.active_syringe_current_zone}' to different target zone '{current_zone}' after insertion.")
                      else:
                           # Moved outside target zones but still visible to manikin cam
                           self._log(f"Active syringe {active_syringe_manikin_id} moved to '{current_zone}' (manikin cam).", level="DEBUG")
                      self.active_syringe_current_zone = current_zone
                 # If still in manikin view, reset the timeout timer implicitly by being here
                 self.last_state_update_time = current_time

            # 3. Check for Timeout: Inserted but not returned
            # Only check timeout if NOT currently seen by manikin cam
            elif not manikin_cam_det:
                 if current_time - self.last_state_update_time > self.INSERT_TO_RETURN_TIMEOUT:
                      self._log_error(self.ERROR_LOST_TRACK, f"Inserted syringe ID {active_syringe_manikin_id} disappeared from manikin view and not seen returning to table within {self.INSERT_TO_RETURN_TIMEOUT:.1f}s.")
                      self._reset_state("Timeout waiting for return to table")
                      return


    def get_log_summary(self) -> str:
        """Returns a summary string of the logged events."""
        return "\n".join(self._log_entries)

# --- End of SyringeTestWorkflow Class ---


# --- Main Execution Script ---
if __name__ == "__main__":

    print("--- Syringe Workflow Test Initializing (Two-Camera Mode) ---")
    print(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    # print(f"Ultralytics Version: {YOLO.version}") # Requires ultralytics 8.1+

    # ----- Configuration: USER MUST EDIT THESE VALUES -----

    # 1. PATH TO YOUR TRAINED YOLO POSE MODEL
    YOLO_MODEL_PATH = "runs/pose/train-pose11n-v30/weights/best.pt" # *** UPDATE THIS PATH ***

    # 2. DEFINE SYRINGE DIAMETERS (in cm)
    POSSIBLE_SYRINGE_DIAMETERS_CM = [0.45, 1.0, 1.25, 1.9] # *** UPDATE THESE VALUES ***

    # 3. CAMERA SOURCES (Use 'webcam' or 'video')
    # If 'webcam', set INDEX. If 'video', set PATH.
    # Ensure video files start at the same time if using 'video'.
    INPUT_TYPE_MANIKIN = 'webcam' # 'webcam' or 'video'
    MANIKIN_CAMERA_INDEX = 0      # Index if webcam
    MANIKIN_VIDEO_PATH = None     # Path if video

    INPUT_TYPE_SYRINGES = 'webcam' # 'webcam' or 'video'
    SYRINGES_CAMERA_INDEX = 1    # Index if webcam (DIFFERENT from manikin)
    SYRINGES_VIDEO_PATH = None   # Path if video

    # 4. FRAME DIMENSIONS (Required if using video files to setup writer correctly)
    #    These are ignored for webcam (obtained dynamically) but needed for video input.
    #    *** SET THESE IF USING VIDEO INPUT ***
    VIDEO_FRAME_WIDTH = 1920
    VIDEO_FRAME_HEIGHT = 1080
    VIDEO_FPS = 30 # Approximate FPS of your video files

    # 5. DEFINE ZONE NAMES AND LOCATIONS PER CAMERA
    #    Names MUST be unique across BOTH cameras.
    #    Coordinates (x1, y1, x2, y2) are relative to EACH camera's frame.

    # --- Manikin Camera Zones ---
    MANIKIN_TARGET_ZONE_NAMES = ["Abdomen", "Head"]
    # *** UPDATE MANIKIN COORDINATES *** (Relative to Manikin Camera Frame)
    MANIKIN_ZONE_DEFINITIONS = [
        ActiveZone(name="Abdomen", rect=(50, 200, 450, 800)),  # Example Left area (adjusted height)
        ActiveZone(name="Head", rect=(VIDEO_FRAME_WIDTH - 450, 200, VIDEO_FRAME_WIDTH - 50, 800)) # Example Right area (adjusted height)
    ]

    # --- Syringe Camera Zones ---
    SYRINGE_TABLE_ZONE_NAMES = ["Table Zone 1", "Table Zone 2", "Table Zone 3", "Table Zone 4"]
    # Add other zones like disposal if needed
    # *** UPDATE SYRINGE COORDINATES *** (Relative to Syringe Camera Frame)
    
    gap = 10  # pixels
    zone_width = 420
    zone_height = 600  # vertical height

    bottom = VIDEO_FRAME_HEIGHT - 50
    top = bottom - zone_height

    SYRINGE_ZONE_DEFINITIONS = [
        ActiveZone(name="Table Zone 1", rect=(100, top, 100 + zone_width, bottom)),
        ActiveZone(name="Table Zone 2", rect=(100 + zone_width + gap, top, 100 + 2 * zone_width + gap, bottom)),
        ActiveZone(name="Table Zone 3", rect=(100 + 2 * (zone_width + gap), top, 100 + 3 * zone_width + 2 * gap, bottom)),
        ActiveZone(name="Table Zone 4", rect=(100 + 3 * (zone_width + gap), top, 100 + 4 * zone_width + 3 * gap, bottom)),
    ]

    # 6. DEFINE THE CORRECT WORKFLOW STEPS FOR THIS TEST RUN
    CORRECT_STARTING_ZONE = "Table Zone 2"      # Must be in SYRINGE_TABLE_ZONE_NAMES
    CORRECT_SYRINGE_DIAMETER_CM = 1.0           # Must be in POSSIBLE_SYRINGE_DIAMETERS_CM
    TARGET_VOLUME_ML = 2.0
    VOLUME_TOLERANCE_ML = 0.75
    CORRECT_TARGET_ZONE = "Abdomen"             # Must be in MANIKIN_TARGET_ZONE_NAMES

    # 7. SAVE OUTPUT VIDEO? (True or False)
    SAVE_OUTPUT_VIDEO = False
    OUTPUT_VIDEO_PATH_MANIKIN = "manikin_processed.mp4"
    OUTPUT_VIDEO_PATH_SYRINGES = "syringes_processed.mp4"

    # 8. OUTPUT FILE NAMES
    RAW_CSV_PATH = 'syringe_volume_data_raw_combined.csv' # Single combined CSV
    WORKFLOW_LOG_PATH = 'syringe_test_workflow_log.txt'

    # 9. DEVICE PREFERENCE (Optional: 'cuda', 'mps', 'cpu', or None for auto)
    #    Set for each estimator if you want specific assignments (e.g., both on GPU, or one on GPU/one CPU)
    DEVICE_PREF_MANIKIN = None # Auto-detect
    DEVICE_PREF_SYRINGES = None # Auto-detect

    # ----- End of User Configuration -----


    # --- Validate Configuration ---
    print("Validating configuration...")
    all_zone_defs = MANIKIN_ZONE_DEFINITIONS + SYRINGE_ZONE_DEFINITIONS
    all_zone_names_defs = [z.name for z in all_zone_defs]
    if len(all_zone_names_defs) != len(set(all_zone_names_defs)):
        print(f"FATAL ERROR: Duplicate zone names found across all defined zones. All names must be unique globally.")
        print(f"Defined names: {all_zone_names_defs}")
        sys.exit(1)

    required_table_zones = set(SYRINGE_TABLE_ZONE_NAMES)
    defined_syringe_zones = set(z.name for z in SYRINGE_ZONE_DEFINITIONS)
    if not required_table_zones.issubset(defined_syringe_zones):
         missing = required_table_zones - defined_syringe_zones
         print(f"FATAL ERROR: The following required table zones are not defined in SYRINGE_ZONE_DEFINITIONS: {missing}")
         sys.exit(1)

    required_target_zones = set(MANIKIN_TARGET_ZONE_NAMES)
    defined_manikin_zones = set(z.name for z in MANIKIN_ZONE_DEFINITIONS)
    if not required_target_zones.issubset(defined_manikin_zones):
         missing = required_target_zones - defined_manikin_zones
         print(f"FATAL ERROR: The following required target zones are not defined in MANIKIN_ZONE_DEFINITIONS: {missing}")
         sys.exit(1)

    if CORRECT_STARTING_ZONE not in required_table_zones:
         print(f"FATAL ERROR: CORRECT_STARTING_ZONE '{CORRECT_STARTING_ZONE}' is not in SYRINGE_TABLE_ZONE_NAMES.")
         sys.exit(1)
    if CORRECT_TARGET_ZONE not in required_target_zones:
         print(f"FATAL ERROR: CORRECT_TARGET_ZONE '{CORRECT_TARGET_ZONE}' is not in MANIKIN_TARGET_ZONE_NAMES.")
         sys.exit(1)
    if CORRECT_SYRINGE_DIAMETER_CM not in POSSIBLE_SYRINGE_DIAMETERS_CM:
         print(f"FATAL ERROR: CORRECT_SYRINGE_DIAMETER_CM '{CORRECT_SYRINGE_DIAMETER_CM}' is not in POSSIBLE_SYRINGE_DIAMETERS_CM.")
         sys.exit(1)
    if (INPUT_TYPE_MANIKIN == 'webcam' and INPUT_TYPE_SYRINGES == 'webcam' and
        MANIKIN_CAMERA_INDEX == SYRINGES_CAMERA_INDEX):
        print(f"FATAL ERROR: Both inputs are set to webcam with the same index ({MANIKIN_CAMERA_INDEX}). Use different indices.")
        sys.exit(1)

    print("Configuration valid.")
    print(f"Possible syringe diameters (cm): {POSSIBLE_SYRINGE_DIAMETERS_CM}")
    print(f"Zone Area Threshold: {0.8 * 100:.0f}%") # Assuming 0.8 default, adjust if needed


    # --- Initialize Components ---
    estimator_manikin = None
    estimator_syringes = None
    workflow = None
    cap_manikin = None
    cap_syringes = None
    out_manikin = None
    out_syringes = None
    csvfile = None
    writer = None

    try:
        # Initialize Estimators
        print("Initializing Manikin Camera Estimator...")
        estimator_manikin = SyringeVolumeEstimator(
            model_path=YOLO_MODEL_PATH,
            possible_diameters_cm=POSSIBLE_SYRINGE_DIAMETERS_CM,
            active_zones=MANIKIN_ZONE_DEFINITIONS,
            device_preference=DEVICE_PREF_MANIKIN
        )

        print("Initializing Syringe Camera Estimator...")
        estimator_syringes = SyringeVolumeEstimator(
            model_path=YOLO_MODEL_PATH,
            possible_diameters_cm=POSSIBLE_SYRINGE_DIAMETERS_CM,
            active_zones=SYRINGE_ZONE_DEFINITIONS,
            device_preference=DEVICE_PREF_SYRINGES
        )

        # Initialize Workflow Manager
        print("Initializing Syringe Test Workflow Manager...")
        workflow = SyringeTestWorkflow(
            table_zone_names=SYRINGE_TABLE_ZONE_NAMES,
            target_zone_names=MANIKIN_TARGET_ZONE_NAMES,
            correct_starting_zone=CORRECT_STARTING_ZONE,
            correct_syringe_diameter=CORRECT_SYRINGE_DIAMETER_CM,
            possible_diameters=POSSIBLE_SYRINGE_DIAMETERS_CM, # Pass diameters
            target_volume_ml=TARGET_VOLUME_ML,
            volume_tolerance_ml=VOLUME_TOLERANCE_ML,
            correct_target_zone=CORRECT_TARGET_ZONE,
            log_file_path=WORKFLOW_LOG_PATH
        )

        # --- Setup Capture Devices ---
        print("Setting up capture devices...")
        # Manikin Camera
        if INPUT_TYPE_MANIKIN == 'video':
            if not MANIKIN_VIDEO_PATH or not os.path.exists(MANIKIN_VIDEO_PATH):
                raise ValueError(f"Manikin video path '{MANIKIN_VIDEO_PATH}' must be provided and exist.")
            cap_manikin = cv2.VideoCapture(MANIKIN_VIDEO_PATH)
            if not cap_manikin.isOpened(): raise IOError(f"Cannot open manikin video file: {MANIKIN_VIDEO_PATH}")
            print(f"Reading Manikin from video: {MANIKIN_VIDEO_PATH}")
            manikin_w = int(cap_manikin.get(cv2.CAP_PROP_FRAME_WIDTH))
            manikin_h = int(cap_manikin.get(cv2.CAP_PROP_FRAME_HEIGHT))
            manikin_fps = cap_manikin.get(cv2.CAP_PROP_FPS) if cap_manikin.get(cv2.CAP_PROP_FPS) > 0 else VIDEO_FPS

        elif INPUT_TYPE_MANIKIN == 'webcam':
            cap_manikin = cv2.VideoCapture(MANIKIN_CAMERA_INDEX)
            if not cap_manikin.isOpened(): raise IOError(f"Cannot open manikin webcam index: {MANIKIN_CAMERA_INDEX}")
            # Optional: Try to set high resolution
            cap_manikin.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
            cap_manikin.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
            manikin_w = int(cap_manikin.get(cv2.CAP_PROP_FRAME_WIDTH))
            manikin_h = int(cap_manikin.get(cv2.CAP_PROP_FRAME_HEIGHT))
            manikin_fps = cap_manikin.get(cv2.CAP_PROP_FPS) if cap_manikin.get(cv2.CAP_PROP_FPS) > 0 else VIDEO_FPS # Default FPS needed
            print(f"Opened Manikin webcam index: {MANIKIN_CAMERA_INDEX} ({manikin_w}x{manikin_h} @ {manikin_fps:.1f} FPS)")
        else:
            raise ValueError(f"Invalid INPUT_TYPE_MANIKIN: '{INPUT_TYPE_MANIKIN}'")

        # Syringe Camera
        if INPUT_TYPE_SYRINGES == 'video':
            if not SYRINGES_VIDEO_PATH or not os.path.exists(SYRINGES_VIDEO_PATH):
                 raise ValueError(f"Syringe video path '{SYRINGES_VIDEO_PATH}' must be provided and exist.")
            cap_syringes = cv2.VideoCapture(SYRINGES_VIDEO_PATH)
            if not cap_syringes.isOpened(): raise IOError(f"Cannot open syringe video file: {SYRINGES_VIDEO_PATH}")
            print(f"Reading Syringes from video: {SYRINGES_VIDEO_PATH}")
            syringes_w = int(cap_syringes.get(cv2.CAP_PROP_FRAME_WIDTH))
            syringes_h = int(cap_syringes.get(cv2.CAP_PROP_FRAME_HEIGHT))
            syringes_fps = cap_syringes.get(cv2.CAP_PROP_FPS) if cap_syringes.get(cv2.CAP_PROP_FPS) > 0 else VIDEO_FPS
            # Sanity check frame rates if using video files
            if abs(manikin_fps - syringes_fps) > 1:
                 print(f"Warning: Video frame rates differ significantly ({manikin_fps:.1f} vs {syringes_fps:.1f}). Synchronization might drift.")

        elif INPUT_TYPE_SYRINGES == 'webcam':
            cap_syringes = cv2.VideoCapture(SYRINGES_CAMERA_INDEX)
            if not cap_syringes.isOpened(): raise IOError(f"Cannot open syringe webcam index: {SYRINGES_CAMERA_INDEX}")
            cap_syringes.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
            cap_syringes.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
            syringes_w = int(cap_syringes.get(cv2.CAP_PROP_FRAME_WIDTH))
            syringes_h = int(cap_syringes.get(cv2.CAP_PROP_FRAME_HEIGHT))
            syringes_fps = cap_syringes.get(cv2.CAP_PROP_FPS) if cap_syringes.get(cv2.CAP_PROP_FPS) > 0 else VIDEO_FPS
            print(f"Opened Syringes webcam index: {SYRINGES_CAMERA_INDEX} ({syringes_w}x{syringes_h} @ {syringes_fps:.1f} FPS)")
        else:
             raise ValueError(f"Invalid INPUT_TYPE_SYRINGES: '{INPUT_TYPE_SYRINGES}'")


        # --- Setup Video Writers (if saving) ---
        if SAVE_OUTPUT_VIDEO:
             estimator_manikin.save_video = True
             estimator_syringes.save_video = True
             print("Setting up video writers...")
             out_manikin = estimator_manikin.setup_video_writer(OUTPUT_VIDEO_PATH_MANIKIN, manikin_w, manikin_h, manikin_fps)
             out_syringes = estimator_syringes.setup_video_writer(OUTPUT_VIDEO_PATH_SYRINGES, syringes_w, syringes_h, syringes_fps)
             if out_manikin is None: estimator_manikin.save_video = False # Disable if setup failed
             if out_syringes is None: estimator_syringes.save_video = False # Disable if setup failed


        # --- Setup CSV Logging (Combined) ---
        print(f"Setting up combined raw data logging to: {RAW_CSV_PATH}")
        os.makedirs(os.path.dirname(RAW_CSV_PATH) or '.', exist_ok=True)
        file_exists = os.path.exists(RAW_CSV_PATH) and os.path.getsize(RAW_CSV_PATH) > 0
        csvfile = open(RAW_CSV_PATH, 'a', newline='')
        writer = csv.writer(csvfile)
        if not file_exists:
            header = ['timestamp', 'camera_source', 'track_id', 'center_x', 'center_y'] + \
                     [f'volume_D{D:.2f}cm' for D in estimator_manikin.possible_diameters] + \
                     ['zone_name', 'in_active_zone_flag']
            writer.writerow(header)

        # --- Main Processing Loop ---
        print("--- Starting Main Loop (Press 'q' in the display window to quit) ---")
        frame_count = 0
        start_time = time.monotonic()

        while True:
            ret_m, frame_m = cap_manikin.read()
            ret_s, frame_s = cap_syringes.read()

            # If either stream ends (especially for video), stop.
            if not ret_m or not ret_s:
                if not ret_m: print("End of Manikin stream or cannot read frame.")
                if not ret_s: print("End of Syringe stream or cannot read frame.")
                break

            current_loop_time = time.monotonic()
            frame_count += 1
            timestamp = current_loop_time # Use monotonic clock for relative time

            # --- Process Frames ---
            # Process Manikin Frame
            annotated_frame_m, detections_m = estimator_manikin.process_frame(
                frame_m, timestamp, writer, camera_source='manikin'
            )
            # Process Syringe Frame
            annotated_frame_s, detections_s = estimator_syringes.process_frame(
                frame_s, timestamp, writer, camera_source='syringes'
            )

            # --- Add camera source info (redundant if passed to process_frame, but safe) ---
            for det in detections_m: det['camera_source'] = 'manikin'
            for det in detections_s: det['camera_source'] = 'syringes'

            # --- Combine Detections for Workflow ---
            all_detections = detections_m + detections_s

            # --- Update Workflow State ---
            workflow.update_state(all_detections, current_loop_time)

            # --- Write Frames to Output Video ---
            if out_manikin is not None and estimator_manikin.save_video:
                 try: out_manikin.write(annotated_frame_m)
                 except Exception as e:
                      print(f"Error writing Manikin video frame: {e}. Disabling saving for Manikin.")
                      out_manikin.release(); out_manikin = None; estimator_manikin.save_video = False
            if out_syringes is not None and estimator_syringes.save_video:
                 try: out_syringes.write(annotated_frame_s)
                 except Exception as e:
                      print(f"Error writing Syringes video frame: {e}. Disabling saving for Syringes.")
                      out_syringes.release(); out_syringes = None; estimator_syringes.save_video = False

            # --- Display Frames ---
            try:
                # Combine frames horizontally for display
                h_m, w_m = annotated_frame_m.shape[:2]
                h_s, w_s = annotated_frame_s.shape[:2]
                max_h = max(h_m, h_s)

                # Resize frames to have the same height
                display_m = cv2.resize(annotated_frame_m, (int(w_m * max_h / h_m), max_h)) if h_m != max_h else annotated_frame_m.copy()
                display_s = cv2.resize(annotated_frame_s, (int(w_s * max_h / h_s), max_h)) if h_s != max_h else annotated_frame_s.copy()

                combined_display = np.hstack((display_m, display_s))

                # Scale down if combined view is too wide
                max_display_width = 1920 # Adjust this based on your screen
                if combined_display.shape[1] > max_display_width:
                    scale = max_display_width / combined_display.shape[1]
                    combined_display = cv2.resize(combined_display, (max_display_width, int(combined_display.shape[0] * scale)))

                cv2.imshow('Syringe Workflow - Manikin (L), Syringes (R) - Press Q', combined_display)
            except Exception as e:
                 print(f"Error displaying frame: {e}")
                 # break # Optional: Stop if display fails

            # --- Exit Condition ---
            key = cv2.waitKey(1) & 0xFF # Wait 1ms
            if key == ord('q'):
                print("'q' pressed, quitting loop.")
                break
            # elif key != 255: print(f"Key pressed: {key}") # Debug other keys

        # --- End of Loop ---
        end_time = time.monotonic()
        print(f"--- Main Loop Ended ---")
        print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
         print(f"FATAL ERROR: File not found - {e}")
         print("Please check paths, especially YOLO_MODEL_PATH and VIDEO_PATHs.")
    except IOError as e:
         print(f"FATAL ERROR: Input/Output error - {e}")
         print("Check camera connections, video file access, file permissions, or camera indices.")
    except ValueError as e:
         print(f"FATAL ERROR: Configuration or Value error - {e}")
         print("Please check the defined parameters (zones, diameters, workflow steps, input types/paths/indices).")
    except ImportError as e:
         print(f"FATAL ERROR: Missing dependency - {e}")
         print("Please ensure all required libraries (ultralytics, torch, opencv-python, numpy) are installed.")
    except torch.cuda.OutOfMemoryError:
         print(f"FATAL ERROR: CUDA Out of Memory.")
         print("Try using CPU for one or both estimators (DEVICE_PREF), using a smaller model, or reducing other GPU load.")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {type(e).__name__} - {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
    finally:
        # --- Cleanup Resources ---
        print("Releasing resources...")
        if cap_manikin is not None and cap_manikin.isOpened():
            cap_manikin.release()
            print("Manikin capture released.")
        if cap_syringes is not None and cap_syringes.isOpened():
            cap_syringes.release()
            print("Syringes capture released.")

        if out_manikin is not None:
            try: out_manikin.release()
            except Exception as e: print(f"Note: Error releasing manikin writer: {e}")
            print(f"Manikin output video writer released. Saved to: {OUTPUT_VIDEO_PATH_MANIKIN}")
        if out_syringes is not None:
            try: out_syringes.release()
            except Exception as e: print(f"Note: Error releasing syringes writer: {e}")
            print(f"Syringes output video writer released. Saved to: {OUTPUT_VIDEO_PATH_SYRINGES}")

        if csvfile is not None and not csvfile.closed:
             csvfile.close()
             print(f"Raw data CSV file closed: {RAW_CSV_PATH}")

        cv2.destroyAllWindows()
        print("Display windows closed.")

        # --- Final Log Summary ---
        if workflow is not None:
            print(f"\n--- Final Workflow Log Summary (also saved to {WORKFLOW_LOG_PATH}) ---")
            print(workflow.get_log_summary())
            print("-------------------------------------------------------------")
        else:
             print("\nWorkflow manager was not initialized, no summary available.")

        print("--- Script Finished ---")