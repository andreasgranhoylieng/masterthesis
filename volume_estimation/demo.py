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
    Handles detection, tracking, and volume estimation of syringes using YOLO Pose.
    Also identifies which ActiveZone a syringe is in.
    """
    def __init__(self, model_path: str, possible_diameters_cm: list[float], active_zones: list[ActiveZone] = None, area_threshold: float = 0.8):
        """
        Initialize the YOLO model, device, ActiveZones, and other parameters.

        Args:
            model_path (str): Path to the trained YOLO Pose model (e.g., 'best.pt').
            possible_diameters_cm (list[float]): List of possible syringe diameters in cm
                                                 that the model should calculate volumes for.
            active_zones (list[ActiveZone], optional): A list of ActiveZone objects. Defaults to [].
            area_threshold (float, optional): The minimum proportion of the syringe's
                                              bounding box area that must be inside an AOI
                                              to consider it 'in' that zone. Defaults to 0.8 (80%).
        """
        self.model_path = model_path
        if not os.path.exists(self.model_path):
             # Attempt a fallback or raise an error
             fallback_path = "yolov8n-pose.pt" # Standard YOLOv8 Nano Pose model
             print(f"Warning: Model not found at '{self.model_path}'.")
             if os.path.exists(fallback_path):
                 print(f"Attempting to use fallback model: '{fallback_path}'")
                 self.model_path = fallback_path
             else:
                 raise FileNotFoundError(f"Specified model '{self.model_path}' not found, and fallback '{fallback_path}' also not found.")

        try:
            self.model = YOLO(self.model_path).eval()
        except Exception as e:
             print(f"Error loading YOLO model from {self.model_path}: {e}")
             print("Please ensure the model path is correct, the file exists, and dependencies are installed.")
             raise

        # Set device based on availability (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # Check for MPS availability properly
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Store possible syringe diameters
        if not isinstance(possible_diameters_cm, list) or not all(isinstance(d, (float, int)) and d > 0 for d in possible_diameters_cm):
             raise ValueError("possible_diameters_cm must be a list of positive numbers.")
        self.possible_diameters = sorted(possible_diameters_cm)
        print(f"Possible syringe diameters (cm): {self.possible_diameters}")

        # --- Active Zone Configuration ---
        self.active_zones = active_zones if active_zones is not None else []
        if not isinstance(self.active_zones, list) or not all(isinstance(zone, ActiveZone) for zone in self.active_zones):
             raise TypeError("active_zones must be a list of ActiveZone objects.")
        zone_names = [zone.name for zone in self.active_zones]
        if len(zone_names) != len(set(zone_names)):
            # Allowing duplicate names might cause ambiguity in zone reporting. Enforcing uniqueness.
            raise ValueError("ActiveZone names must be unique.")
        self.area_threshold = max(0.1, min(1.0, area_threshold)) # Clamp threshold between 0.1 and 1.0
        print(f"Active Zones defined: {[zone.name for zone in self.active_zones]}")
        print(f"Zone Area Threshold: {self.area_threshold * 100:.0f}%")
        # --- End Active Zone Configuration ---

        # Internal state for FPS calculation
        self.last_timestamps = deque(maxlen=10) # Use more samples for smoother FPS
        self.inference_size = None # Let YOLO handle unless cropping
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


    def draw_fps_counter(self, frame: np.ndarray) -> None:
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

                text = f"FPS: {avg_fps:.1f}"
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


    def process_frame(self, frame: np.ndarray, timestamp: float, writer: csv.writer = None) -> tuple[np.ndarray, list]:
        """
        Process a single frame: detect, track, check zones, estimate volumes, draw, and log.

        Args:
            frame (np.ndarray): The input video frame.
            timestamp (float): The timestamp associated with the frame (e.g., seconds).
            writer (csv.writer, optional): CSV writer object to log raw data. Defaults to None.

        Returns:
            tuple:
                - annotated_frame (np.ndarray): Frame with drawings (zones, detections, tables, FPS).
                - detections (list[dict]): List of detected syringes with their info:
                  Each dict contains: 'id', 'zone', 'volumes', 'bbox', 'center', 'in_active_zone_flag'.
                  Returns an empty list if no valid syringes are detected or tracking fails.
        """
        original_frame_height, original_frame_width = frame.shape[:2]
        annotated_frame = frame.copy() # Work on a copy for drawing

        # --- Draw Active Zones ---
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
        self.draw_fps_counter(annotated_frame)

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
            print(f"Error during YOLO tracking/plotting: {e}")
            # Log an error row if writer is available
            if writer:
                 row = [timestamp, np.nan, np.nan, np.nan] + [np.nan for _ in self.possible_diameters] + ['TrackingError', 0]
                 writer.writerow(row)
            return annotated_frame, [] # Return frame with zones/fps, but no detections


        # --- Process each detected/tracked object ---
        # Check if tracker returned boxes and IDs
        if results_data.boxes is None or results_data.boxes.id is None:
             # No tracks found in this frame
             return annotated_frame, []

        # Iterate through detected boxes and their track IDs
        for i, box in enumerate(results_data.boxes):
            track_id = int(box.id[0]) # Get track ID

            # --- Bounding Box and Center ---
            box_coords = box.xyxy[0].numpy() # Get box coordinates (x1, y1, x2, y2)
            x1_syr, y1_syr, x2_syr, y2_syr = map(int, box_coords)
            if x1_syr >= x2_syr or y1_syr >= y2_syr: continue # Skip invalid boxes
            syringe_bbox = (x1_syr, y1_syr, x2_syr, y2_syr)
            center_x = float((x1_syr + x2_syr) / 2)
            center_y = float((y1_syr + y2_syr) / 2)
            syringe_box_area = float((x2_syr - x1_syr) * (y2_syr - y1_syr))

            # --- Zone Detection ---
            in_active_zone_flag = False
            detected_zone_name = "Outside" # Default
            if syringe_box_area > 0:
                best_overlap_zone = None
                max_overlap_ratio = 0.0
                for zone in self.active_zones:
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
                    # Keypoints order expected: 0:bottom-left, 1:top-left, 2:top-right, 3:bottom-right (adjust if model differs)
                    kpts = results_data.keypoints.xy[i][:4].numpy() # Get first 4 keypoints

                    # Check keypoint confidence if available (optional)
                    # kpts_conf = results_data.keypoints.conf[i][:4].numpy()
                    # min_kpt_conf = 0.5 # Example threshold
                    # if np.any(kpts_conf < min_kpt_conf):
                    #     raise ValueError("Low keypoint confidence")

                    ll_point, ul_point, ur_point, lr_point = kpts

                    # Calculate average width and height in pixels from keypoints
                    width_px = (np.linalg.norm(lr_point - ll_point) + np.linalg.norm(ur_point - ul_point)) / 2.0
                    height_px = (np.linalg.norm(ul_point - ll_point) + np.linalg.norm(ur_point - lr_point)) / 2.0

                    # Estimate volume only if width and height seem valid
                    if width_px > 2 and height_px > 2: # Basic sanity check (at least a few pixels)
                        for D_cm in self.possible_diameters:
                            # Calculate scale factor (cm per pixel) based on this diameter assumption
                            scale_factor_cm_per_px = D_cm / width_px
                            # Estimate liquid column height in cm
                            H_cm = height_px * scale_factor_cm_per_px

                            # Validate calculated height (e.g., 0 to 25 cm is plausible for syringe liquid)
                            if 0 < H_cm <= 25.0:
                                radius_cm = D_cm / 2.0
                                volume_ml = math.pi * (radius_cm ** 2) * H_cm
                                volumes_for_diameters[D_cm] = volume_ml
                            # else: volume remains NaN if H_cm is invalid

                except Exception as e:
                    # Silently ignore volume calculation errors (e.g., low confidence, math error)
                    # print(f"Debug: Volume calculation error for ID {track_id}: {e}") # Optional debug print
                    pass # Keep volumes as NaN

            # --- Store Results for this Syringe ---
            detection_info = {
                'id': track_id,
                'zone': detected_zone_name,
                'volumes': volumes_for_diameters, # Dict {diameter: volume_ml}
                'bbox': syringe_bbox,             # Tuple (x1, y1, x2, y2)
                'center': (center_x, center_y),   # Tuple (cx, cy)
                'in_active_zone_flag': in_active_zone_flag # Boolean
            }
            processed_detections.append(detection_info)

            # --- Log Raw Data to CSV (if writer provided) ---
            if writer:
                volumes_list = [volumes_for_diameters.get(D, np.nan) for D in self.possible_diameters]
                log_row = [timestamp, track_id, center_x, center_y] + volumes_list + \
                          [detected_zone_name, 1 if in_active_zone_flag else 0]
                try:
                    writer.writerow(log_row)
                except Exception as e:
                    print(f"Error writing row to CSV: {e}")


            # --- Draw Volume Table ---
            # Convert dict to list of tuples for drawing function
            volumes_for_table = list(volumes_for_diameters.items())
            table_x = x2_syr + 10 # Position right of the bounding box
            table_y = y1_syr      # Align with top of the bounding box
            self.draw_volume_table(annotated_frame, volumes_for_table, table_x, table_y, track_id)


        # --- Post-processing Drawing ---
        # Redraw boxes in green for syringes detected inside *any* active zone
        for det in processed_detections:
             if det['in_active_zone_flag']:
                 x1, y1, x2, y2 = map(int, det['bbox'])
                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green outline, thickness 2

        # Draw overall "ACTIVE ZONE DETECTED" banner if needed
        if any(d['in_active_zone_flag'] for d in processed_detections):
            text = "ACTIVE ZONE DETECTED"
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


    def setup_capture(self, input_source: str ='webcam', video_path: str =None) -> cv2.VideoCapture:
        """Sets up the video capture object based on the source."""
        cap = None
        if input_source == 'video':
            if not video_path or not os.path.exists(video_path):
                raise ValueError(f"Video path '{video_path}' must be provided and exist for input_source='video'")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                 raise IOError(f"Cannot open video file: {video_path}")
            print(f"Reading from video: {video_path}")
        elif input_source == 'webcam':
            # Try different camera indices if 0 doesn't work
            for cam_index in range(3): # Try 0, 1, 2
                cap = cv2.VideoCapture(cam_index)
                if cap.isOpened():
                    print(f"Opened webcam with index: {cam_index}")
                    break
                else:
                    cap.release() # Release if not opened successfully
            if cap is None or not cap.isOpened():
                raise IOError("Cannot open webcam. Tried indices 0, 1, 2.")

            # Optional: Try to set resolution (camera might ignore it)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Reading from webcam (Resolution: {actual_width}x{actual_height})")
        else:
             raise ValueError(f"Invalid input_source: '{input_source}'. Choose 'webcam' or 'video'.")

        return cap


    def setup_video_writer(self, output_path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        """Sets up the video writer object if save_video is True."""
        if not self.save_video:
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: # Handle case where FPS is not available or zero
             fps = 25.0 # Use a common default FPS
             print(f"Warning: Could not get valid FPS from source ({cap.get(cv2.CAP_PROP_FPS)}). Using default: {fps:.1f} FPS")

        # Use frame dimensions directly from the capture device
        write_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        write_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if write_width <= 0 or write_height <= 0:
             print(f"Error: Invalid frame dimensions from capture ({write_width}x{write_height}). Cannot create video writer.")
             self.save_video = False # Disable saving
             return None

        # Choose a common codec (mp4v for .mp4, XVID for .avi)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            out = cv2.VideoWriter(output_path, fourcc, fps, (write_width, write_height))
            if not out.isOpened():
                 raise IOError(f"VideoWriter failed to open for path: {output_path}")
            print(f"💾 Saving processed video to: {output_path} ({write_width}x{write_height} @ {fps:.1f} FPS)")
            return out
        except Exception as e:
            print(f"Error initializing VideoWriter: {e}. Video saving disabled.")
            self.save_video = False
            return None
# --- End of SyringeVolumeEstimator Class ---


# --- Define the Syringe Test Workflow Class ---
class SyringeTestWorkflow:
    """Manages and validates a multi-step syringe handling test."""

    # Define states
    STATE_IDLE = "IDLE"                     # Waiting for the correct syringe to be picked
    STATE_SYRINGE_PICKED = "SYRINGE_PICKED" # Correct syringe is being handled (outside table)
    STATE_SYRINGE_INSERTED = "SYRINGE_INSERTED" # Syringe detected in a target zone

    # Define error types
    ERROR_WRONG_SYRINGE = "Wrong Syringe Picked"
    ERROR_WRONG_VOLUME = "Incorrect Volume"
    ERROR_WRONG_TARGET = "Wrong Target Zone"
    ERROR_MULTI_ACTIVE = "Multiple Syringes Active"
    ERROR_LOST_TRACK = "Lost Track of Syringe"
    ERROR_PREMATURE_RETURN = "Syringe Returned Prematurely"
    ERROR_VOLUME_UNDETERMINED = "Volume Undetermined"


    def __init__(self,
                 table_zone_names: list[str],
                 target_zone_names: list[str],
                 correct_starting_zone: str,
                 correct_syringe_diameter: float,
                 possible_diameters: list[float], # Pass from estimator
                 target_volume_ml: float,
                 volume_tolerance_ml: float,
                 correct_target_zone: str,
                 log_file_path: str = "workflow_log.txt"):
        """
        Initializes the workflow manager.
        # ... (Args documentation remains the same) ...
        """
        self.table_zones = set(table_zone_names)
        self.target_zones = set(target_zone_names)
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

        # --- State Variables ---  <<<<< MOVED THIS BLOCK UP
        self.current_state = self.STATE_IDLE
        self.active_syringe_id = None            # Track ID of the syringe being handled
        self.active_syringe_start_zone = None    # Zone the active syringe originated from
        self.active_syringe_current_zone = None  # Most recent zone of the active syringe
        self.active_syringe_volume = None        # Measured volume at insertion time
        self.error_flags_this_cycle = set()      # Store errors encountered in the current pickup-insert-return cycle
        self.last_state_update_time = time.monotonic() # Track time for timeouts

        # --- Tracking History (Improved Logic) ---
        self.syringe_last_known_zone = defaultdict(lambda: "Unknown") # {track_id: zone_name}
        self.syringe_last_seen_time = defaultdict(float)             # {track_id: timestamp}
        self.SYRINGE_PURGE_TIMEOUT = 15.0 # Seconds after which info about a missing syringe is purged

        # --- Logging Setup ---   <<<<< NOW STATE IS INITIALIZED BEFORE LOGGING STARTS
        self.log_file_path = log_file_path
        self._log_entries = [] # Store logs in memory during run
        self._clear_log_file() # Clear previous log file on initialization
        self._log_header()     # Log the configuration details (Now safe to call)


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
        self._log("--- Workflow Initialized ---", level="SETUP")
        self._log(f"Table Zones: {list(self.table_zones)}", level="SETUP")
        self._log(f"Target Zones: {list(self.target_zones)}", level="SETUP")
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
                 summary = f"Cycle for syringe ID {self.active_syringe_id} ended."
                 if self.error_flags_this_cycle:
                     summary += f" Errors: {', '.join(sorted(list(self.error_flags_this_cycle)))}."
                 else:
                     summary += " Status: OK."
                 self._log(summary, level="SUMMARY")

            self._log(f"Resetting state to IDLE. Reason: {reason}")

        self.current_state = self.STATE_IDLE
        self.active_syringe_id = None
        self.active_syringe_start_zone = None
        self.active_syringe_current_zone = None
        self.active_syringe_volume = None
        self.error_flags_this_cycle = set() # Clear errors for the new cycle


    def _purge_old_tracks(self, current_time: float):
         """Removes information about tracks not seen for a while."""
         purged_ids = [
             track_id for track_id, last_seen in self.syringe_last_seen_time.items()
             if current_time - last_seen > self.SYRINGE_PURGE_TIMEOUT
         ]
         for track_id in purged_ids:
             del self.syringe_last_known_zone[track_id]
             del self.syringe_last_seen_time[track_id]
             # Optional: Log purge event if needed for debugging
             # self._log(f"Purged tracking info for inactive ID {track_id}", level="DEBUG")


    def update_state(self, detections: list[dict], current_time: float):
        """
        Updates the workflow state based on the latest syringe detections.

        Args:
            detections (list[dict]): List of detection dictionaries from process_frame.
            current_time (float): Current timestamp (e.g., from time.monotonic()).
        """
        self.last_state_update_time = current_time
        current_detections_map = {det['id']: det for det in detections if det['id'] != -1} # Map ID -> detection info

        # --- Update Tracking History ---
        for track_id, det in current_detections_map.items():
             self.syringe_last_known_zone[track_id] = det['zone']
             self.syringe_last_seen_time[track_id] = current_time

        # --- Purge Old Tracks ---
        self._purge_old_tracks(current_time)

        # --- Identify Active Syringe(s) ---
        # An active syringe is one whose last known zone is NOT a table zone or "Unknown".
        active_syringes = []
        for track_id, zone in self.syringe_last_known_zone.items():
            # Consider a syringe active if it's currently detected outside a table zone,
            # OR if it was previously active and hasn't been seen back on the table yet.
            is_currently_outside_table = track_id in current_detections_map and current_detections_map[track_id]['zone'] not in self.table_zones
            is_persisted_active = track_id == self.active_syringe_id and zone not in self.table_zones

            if is_currently_outside_table or is_persisted_active:
                 # Get the most recent detection info if available
                 current_det = current_detections_map.get(track_id)
                 if current_det:
                    active_syringes.append(current_det)
                 elif is_persisted_active :
                      # Syringe was active but not detected this frame, keep tracking it briefly
                      # Create a placeholder detection using last known info
                      placeholder_det = {
                           'id': track_id, 'zone': self.syringe_last_known_zone[track_id],
                           'volumes': {}, 'bbox': None, 'center': None, 'in_active_zone_flag': False
                           }
                      active_syringes.append(placeholder_det)


        # --- State Machine Logic ---
        active_detection = None # The single syringe we focus on

        if len(active_syringes) == 1:
            active_detection = active_syringes[0]
            # If we just picked up a syringe, ensure it's the one we were tracking
            if self.current_state != self.STATE_IDLE and active_detection['id'] != self.active_syringe_id:
                self._log_error(self.ERROR_WRONG_SYRINGE, f"Unexpected syringe ID {active_detection['id']} became active (expected {self.active_syringe_id}).")
                self._reset_state("Unexpected syringe activity")
                return # Stop processing this frame

        elif len(active_syringes) > 1:
            # Error condition: More than one syringe seems active
            active_ids = [d['id'] for d in active_syringes]
            if self.current_state != self.STATE_IDLE: # Only error if already in an active state
                 self._log_error(self.ERROR_MULTI_ACTIVE, f"Multiple syringes ({active_ids}) detected outside table zones.")
                 self._reset_state("Multiple active syringes detected")
            # else: Could be initial state noise, ignore for now
            return # Skip further processing this frame

        # --- Handle case where the active syringe disappears ---
        TIMEOUT_ACTIVE_SECONDS = 5.0 # How long to wait before declaring lost track
        if self.current_state != self.STATE_IDLE and active_detection is None:
            # No active syringe detected, was one being tracked?
            if self.active_syringe_id is not None:
                # Check if it reappeared on the table
                if self.syringe_last_known_zone[self.active_syringe_id] in self.table_zones:
                     returned_zone = self.syringe_last_known_zone[self.active_syringe_id]
                     if self.current_state == self.STATE_SYRINGE_PICKED:
                          self._log_error(self.ERROR_PREMATURE_RETURN, f"Syringe ID {self.active_syringe_id} returned to table zone '{returned_zone}' before insertion.")
                     elif self.current_state == self.STATE_SYRINGE_INSERTED:
                           self._log(f"Syringe ID {self.active_syringe_id} returned to table zone '{returned_zone}'.")
                     self._reset_state(f"Syringe returned to table zone '{returned_zone}'")

                # Check for timeout if not seen on table
                elif current_time - self.syringe_last_seen_time.get(self.active_syringe_id, current_time) > TIMEOUT_ACTIVE_SECONDS:
                     self._log_error(self.ERROR_LOST_TRACK, f"Track lost for active syringe ID {self.active_syringe_id} for > {TIMEOUT_ACTIVE_SECONDS:.1f}s.")
                     self._reset_state("Active syringe lost track (timeout)")
            return # No active syringe to process


        # --- Process based on Current State ---

        # == STATE: IDLE ==
        if self.current_state == self.STATE_IDLE:
            if active_detection:
                # A syringe just became active. Check if it came from the correct starting zone.
                picked_id = active_detection['id']
                # Determine origin zone (requires history - check where it WAS before becoming active)
                origin_zone = "Unknown"
                # Check previous frames' history (simplification: assume previous zone was table if now active)
                # A more robust check would query history for the frame *before* it left the table.
                # For now, we rely on the _assumption_ that the first picked syringe IS the target one.
                # We check if it *could* have come from the correct zone.
                potential_origin_zone = self.syringe_last_known_zone[picked_id] # Where was it last seen?

                # Log the pickup event
                self._log(f"Syringe ID {picked_id} became active. Last seen in zone '{potential_origin_zone}'. State -> {self.STATE_SYRINGE_PICKED}.")

                # Assign active syringe info
                self.active_syringe_id = picked_id
                self.active_syringe_start_zone = potential_origin_zone # Store the potential origin
                self.active_syringe_current_zone = active_detection['zone']
                self.current_state = self.STATE_SYRINGE_PICKED
                self.error_flags_this_cycle = set() # Start fresh error set for this cycle

                # Check if the potential origin was the WRONG starting zone
                # This check is weak if the syringe was 'Unknown' or 'Outside' before pickup.
                if potential_origin_zone != "Unknown" and potential_origin_zone != self.correct_starting_zone:
                     # Only flag if we are reasonably sure it came from a specific *wrong* table zone.
                     if potential_origin_zone in self.table_zones:
                          self._log_error(self.ERROR_WRONG_SYRINGE, f"Picked from '{potential_origin_zone}', expected '{self.correct_starting_zone}'.")

        # == STATE: SYRINGE_PICKED ==
        elif self.current_state == self.STATE_SYRINGE_PICKED:
            if active_detection and active_detection['id'] == self.active_syringe_id:
                # Still tracking the correct syringe
                current_zone = active_detection['zone']
                self.active_syringe_current_zone = current_zone

                # Check if inserted into a TARGET zone
                if current_zone in self.target_zones:
                    self._log(f"Syringe ID {self.active_syringe_id} entered target zone '{current_zone}'. State -> {self.STATE_SYRINGE_INSERTED}.")
                    self.current_state = self.STATE_SYRINGE_INSERTED

                    # --- Perform Insertion Checks ---
                    # 1. Check Volume: Retrieve volume for the CORRECT diameter
                    volume = active_detection['volumes'].get(self.correct_diameter, float('nan'))
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
                        self._log_error(self.ERROR_VOLUME_UNDETERMINED, f"Could not determine valid volume for diameter {self.correct_diameter}cm.")

                    # 2. Check Target Zone
                    if current_zone != self.correct_target_zone:
                        self._log_error(self.ERROR_WRONG_TARGET, f"Inserted into WRONG zone '{current_zone}'. Expected '{self.correct_target_zone}'.")
                    else:
                         self._log(f"Inserted into CORRECT target zone '{current_zone}'.")

                    # 3. Re-check starting zone (if we logged an error at pickup)
                    if self.ERROR_WRONG_SYRINGE in self.error_flags_this_cycle:
                        self._log("Reminder: Syringe was potentially picked from the wrong starting zone.", level="WARN")


                # Check if returned to a TABLE zone prematurely
                elif current_zone in self.table_zones:
                     self._log_error(self.ERROR_PREMATURE_RETURN, f"Syringe ID {self.active_syringe_id} returned to table zone '{current_zone}' before insertion.")
                     self._reset_state(f"Syringe returned to table zone '{current_zone}' prematurely")

            # (Case where active syringe disappears is handled by the timeout logic above)

        # == STATE: SYRINGE_INSERTED ==
        elif self.current_state == self.STATE_SYRINGE_INSERTED:
             # Waiting for the syringe to be returned to a table zone
             if active_detection and active_detection['id'] == self.active_syringe_id:
                 # Syringe still detected, check its current zone
                 current_zone = active_detection['zone']
                 if current_zone in self.table_zones:
                     self._log(f"Syringe ID {self.active_syringe_id} returned to table zone '{current_zone}'.")
                     self._reset_state(f"Workflow cycle completed, syringe returned to '{current_zone}'")
                 elif current_zone == "Outside":
                      # Still moving, potentially leaving target zone
                      self._log(f"Syringe ID {self.active_syringe_id} moved to 'Outside' zone after insertion.", level="DEBUG")
                 elif current_zone in self.target_zones and current_zone != self.active_syringe_current_zone:
                      # Moved to a *different* target zone after initial insertion?
                      self._log_error(self.ERROR_WRONG_TARGET, f"Syringe moved from '{self.active_syringe_current_zone}' to different target zone '{current_zone}' after insertion.")
                      self.active_syringe_current_zone = current_zone # Update current zone

             # (Case where active syringe disappears is handled by the timeout logic above, leading to reset)


    def get_log_summary(self) -> str:
        """Returns a summary string of the logged events."""
        return "\n".join(self._log_entries)

# --- End of SyringeTestWorkflow Class ---


# --- Main Execution Script ---
if __name__ == "__main__":

    print("--- Syringe Workflow Test Initializing ---")
    print(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    #print(f"Ultralytics Version: {YOLO().version}") # Get YOLO version

    # ----- Configuration: USER MUST EDIT THESE VALUES -----

    # 1. PATH TO YOUR TRAINED YOLO POSE MODEL
    #    (e.g., from a 'runs/pose/trainXX/weights' directory)
    #    If this path is wrong, it will try to use 'yolov8n-pose.pt' as a fallback.
    YOLO_MODEL_PATH = "runs/pose/train-pose11n-v30/weights/best.pt" # *** UPDATE THIS PATH ***

    # 2. DEFINE SYRINGE DIAMETERS (in cm) that your model estimates volume for.
    #    These *must* match the diameters the Volume Estimator uses.
    POSSIBLE_SYRINGE_DIAMETERS_CM = [0.45, 1.0, 1.25, 1.9] # Example: 1ml, 5ml, 10ml, 20ml syringes - *** UPDATE THESE VALUES ***

    # 3. DEFINE ZONE NAMES AND LOCATIONS
    #    Names MUST be unique. Coordinates are (x1, y1, x2, y2) relative to the camera frame.
    #    Use a tool/script to find the correct pixel coordinates for your setup.
    FRAME_WIDTH = 1920 # Example width - Set based on your camera resolution
    FRAME_HEIGHT = 1080 # Example height - Set based on your camera resolution

    TABLE_ZONE_NAMES = ["Table Zone 1", "Table Zone 2", "Table Zone 3", "Table Zone 4"]
    TARGET_ZONE_NAMES = ["Abdomen", "Head"]

    # Example Zone Layout (*** UPDATE THESE COORDINATES ***)
    ZONE_DEFINITIONS = [
        # Table Zones (Bottom of frame example)
        ActiveZone(name="Table Zone 1", rect=(100, FRAME_HEIGHT - 250, 300, FRAME_HEIGHT - 50)),
        ActiveZone(name="Table Zone 2", rect=(350, FRAME_HEIGHT - 250, 550, FRAME_HEIGHT - 50)),
        ActiveZone(name="Table Zone 3", rect=(600, FRAME_HEIGHT - 250, 800, FRAME_HEIGHT - 50)),
        ActiveZone(name="Table Zone 4", rect=(850, FRAME_HEIGHT - 250, 1050, FRAME_HEIGHT - 50)),
        # Target Zones (Left/Right side example)
        ActiveZone(name="Abdomen", rect=(50, 200, 450, FRAME_HEIGHT - 300)),  # Left area
        ActiveZone(name="Head", rect=(FRAME_WIDTH - 450, 200, FRAME_WIDTH - 50, FRAME_HEIGHT - 300)) # Right area
    ]

    # 4. DEFINE THE CORRECT WORKFLOW STEPS FOR THIS TEST RUN
    CORRECT_STARTING_ZONE = "Table Zone 2"      # Name of the zone where the correct syringe starts
    CORRECT_SYRINGE_DIAMETER_CM = 1.0       # Diameter (cm) of the syringe used in THIS test (must be in POSSIBLE_SYRINGE_DIAMETERS_CM)
    TARGET_VOLUME_ML = 2.0                      # Target fill volume (mL)
    VOLUME_TOLERANCE_ML = 0.75                  # Allowed deviation (+/- mL) from target volume
    CORRECT_TARGET_ZONE = "Abdomen"             # Name of the correct insertion zone

    # 5. CHOOSE INPUT SOURCE ('webcam' or 'video')
    INPUT_SOURCE = 'webcam'
    # INPUT_SOURCE = 'video' # Uncomment and set VIDEO_PATH if using video file

    # 6. VIDEO FILE PATH (only used if INPUT_SOURCE is 'video')
    VIDEO_PATH = None # Example: 'path/to/your/test_video.mp4' # *** SET VIDEO PATH if needed ***

    # 7. SAVE OUTPUT VIDEO? (True or False)
    #    Processed video will be saved next to the input video or as 'webcam_processed_workflow.mp4'
    SAVE_OUTPUT_VIDEO = False

    # 8. OUTPUT FILE NAMES
    RAW_CSV_PATH = 'syringe_volume_data_raw.csv'  # Logs raw volume estimations per frame
    WORKFLOW_LOG_PATH = 'syringe_test_workflow_log.txt' # Logs test steps and errors

    # ----- End of User Configuration -----


    # --- Validate Configuration ---
    if not ZONE_DEFINITIONS:
         print("FATAL ERROR: ZONE_DEFINITIONS list is empty. Please define ActiveZone objects.")
         sys.exit(1)
    if CORRECT_STARTING_ZONE not in TABLE_ZONE_NAMES:
         print(f"FATAL ERROR: CORRECT_STARTING_ZONE '{CORRECT_STARTING_ZONE}' is not in TABLE_ZONE_NAMES.")
         sys.exit(1)
    if CORRECT_TARGET_ZONE not in TARGET_ZONE_NAMES:
         print(f"FATAL ERROR: CORRECT_TARGET_ZONE '{CORRECT_TARGET_ZONE}' is not in TARGET_ZONE_NAMES.")
         sys.exit(1)
    if CORRECT_SYRINGE_DIAMETER_CM not in POSSIBLE_SYRINGE_DIAMETERS_CM:
         print(f"FATAL ERROR: CORRECT_SYRINGE_DIAMETER_CM '{CORRECT_SYRINGE_DIAMETER_CM}' is not in POSSIBLE_SYRINGE_DIAMETERS_CM.")
         sys.exit(1)
    zone_names_from_defs = [z.name for z in ZONE_DEFINITIONS]
    if len(zone_names_from_defs) != len(set(zone_names_from_defs)):
        print(f"FATAL ERROR: Duplicate names found in ZONE_DEFINITIONS. All names must be unique.")
        sys.exit(1)
    required_zones = set(TABLE_ZONE_NAMES + TARGET_ZONE_NAMES)
    defined_zones = set(zone_names_from_defs)
    if not required_zones.issubset(defined_zones):
         missing = required_zones - defined_zones
         print(f"FATAL ERROR: The following required zones are not defined in ZONE_DEFINITIONS: {missing}")
         sys.exit(1)


    # --- Initialize Components ---
    estimator = None
    workflow = None
    cap = None
    out = None
    csvfile = None
    writer = None

    try:
        # Initialize Estimator
        print("Initializing Syringe Volume Estimator...")
        estimator = SyringeVolumeEstimator(
            model_path=YOLO_MODEL_PATH,
            possible_diameters_cm=POSSIBLE_SYRINGE_DIAMETERS_CM,
            active_zones=ZONE_DEFINITIONS,
            area_threshold=0.8 # 80% overlap needed to be 'in' a zone
        )

        # Initialize Workflow Manager
        print("Initializing Syringe Test Workflow Manager...")
        workflow = SyringeTestWorkflow(
            table_zone_names=TABLE_ZONE_NAMES,
            target_zone_names=TARGET_ZONE_NAMES,
            correct_starting_zone=CORRECT_STARTING_ZONE,
            correct_syringe_diameter=CORRECT_SYRINGE_DIAMETER_CM,
            possible_diameters=estimator.possible_diameters, # Pass diameters from estimator
            target_volume_ml=TARGET_VOLUME_ML,
            volume_tolerance_ml=VOLUME_TOLERANCE_ML,
            correct_target_zone=CORRECT_TARGET_ZONE,
            log_file_path=WORKFLOW_LOG_PATH
        )

        # Setup capture device
        print(f"Setting up input source: {INPUT_SOURCE}...")
        cap = estimator.setup_capture(input_source=INPUT_SOURCE, video_path=VIDEO_PATH)

        # Setup video writer if saving
        output_video_path = None
        if SAVE_OUTPUT_VIDEO:
             estimator.save_video = True # Enable saving flag in estimator
             if INPUT_SOURCE == 'video':
                  if VIDEO_PATH: # Ensure video path is not None
                    base, ext = os.path.splitext(VIDEO_PATH)
                    output_video_path = f"{base}_processed_workflow{ext}"
                  else:
                    print("Warning: SAVE_OUTPUT_VIDEO is True for video input, but VIDEO_PATH is not set. Saving disabled.")
                    estimator.save_video = False
             else: # Webcam output name
                  output_video_path = "webcam_processed_workflow.mp4"

             if estimator.save_video and output_video_path:
                  out = estimator.setup_video_writer(output_video_path, cap)
                  if out is None: # setup_video_writer handles errors and might disable saving
                      estimator.save_video = False
                      print("Video saving failed to initialize.")
        else:
             estimator.save_video = False


        # --- Setup CSV Logging ---
        print(f"Setting up raw data logging to: {RAW_CSV_PATH}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(RAW_CSV_PATH) or '.', exist_ok=True)
        # Check if file needs header
        file_exists = os.path.exists(RAW_CSV_PATH) and os.path.getsize(RAW_CSV_PATH) > 0
        csvfile = open(RAW_CSV_PATH, 'a', newline='')
        writer = csv.writer(csvfile)
        if not file_exists:
            header = ['timestamp', 'track_id', 'center_x', 'center_y'] + \
                     [f'volume_D{D:.2f}cm' for D in estimator.possible_diameters] + \
                     ['zone_name', 'in_active_zone_flag']
            writer.writerow(header)

        # --- Main Processing Loop ---
        print("--- Starting Main Loop (Press 'q' in the display window to quit) ---")
        frame_count = 0
        start_time = time.monotonic()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or cannot read frame.")
                break

            current_loop_time = time.monotonic()
            frame_count += 1

            # Determine timestamp for logging (use monotonic clock for intervals)
            timestamp = current_loop_time # Relative time in seconds from start

            # Process frame using the estimator to get detections and annotated frame
            annotated_frame, detections = estimator.process_frame(frame, timestamp, writer)

            # Update the workflow state machine with the detections
            workflow.update_state(detections, current_loop_time)

            # Write frame to output video if enabled and writer is valid
            if out is not None and estimator.save_video:
                 try:
                     out.write(annotated_frame)
                 except Exception as e:
                      print(f"Error writing video frame: {e}. Disabling further saving.")
                      out.release() # Release the faulty writer
                      out = None
                      estimator.save_video = False # Prevent further attempts

            # Display the annotated frame
            display_scale = 0.7 # Scale down large frames for display (adjust as needed)
            try:
                 if annotated_frame.shape[1] > 1920 or annotated_frame.shape[0] > 1080:
                      display_height = int(annotated_frame.shape[0] * display_scale)
                      display_width = int(annotated_frame.shape[1] * display_scale)
                      annotated_frame_display = cv2.resize(annotated_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                 else:
                      annotated_frame_display = annotated_frame

                 cv2.imshow('Syringe Test Workflow - Press Q to Quit', annotated_frame_display)
            except Exception as e:
                 print(f"Error displaying frame: {e}")
                 # Decide if you want to break or continue if display fails
                 # break

            # Exit condition: Press 'q'
            key = cv2.waitKey(1) & 0xFF # Wait 1ms for key press
            if key == ord('q'):
                print("'q' pressed, quitting loop.")
                break
            elif key != 255: # 255 is returned when no key is pressed
                 print(f"Key pressed: {key}") # Log other key presses if needed

        # --- End of Loop ---
        end_time = time.monotonic()
        print(f"--- Main Loop Ended ---")
        print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
         print(f"FATAL ERROR: File not found - {e}")
         print("Please check paths, especially YOLO_MODEL_PATH and VIDEO_PATH.")
    except IOError as e:
         print(f"FATAL ERROR: Input/Output error - {e}")
         print("Check camera connection, video file access, or file permissions.")
    except ValueError as e:
         print(f"FATAL ERROR: Configuration or Value error - {e}")
         print("Please check the defined parameters (zones, diameters, workflow steps).")
    except ImportError as e:
         print(f"FATAL ERROR: Missing dependency - {e}")
         print("Please ensure all required libraries (ultralytics, torch, opencv-python, numpy) are installed.")
    except torch.cuda.OutOfMemoryError:
         print(f"FATAL ERROR: CUDA Out of Memory.")
         print("Try reducing batch size (if applicable), using a smaller model, or running on CPU if GPU memory is insufficient.")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        print("--- Traceback ---")
        traceback.print_exc() # Print detailed traceback for debugging
        print("-----------------")
    finally:
        # --- Cleanup Resources ---
        print("Releasing resources...")
        if cap is not None and cap.isOpened():
            cap.release()
            print("Video capture released.")
        if out is not None and estimator is not None and estimator.save_video:
             # Check if writer is still open before releasing
             # (It might have been closed/released due to write errors)
            try:
                 # A simple check; ideally VideoWriter would have an isOpened() method
                 if out.get(cv2.CAP_PROP_FOURCC) != 0: # Check if FourCC is still set
                      out.release()
                      print(f"Output video writer released. Saved to: {output_video_path}")
            except Exception as e:
                 print(f"Note: Error during final video writer release (might be already closed): {e}")

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