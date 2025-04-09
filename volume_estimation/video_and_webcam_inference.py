import csv
import math
import os
import time
from collections import deque

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
        """
        if not isinstance(name, str) or not name:
            raise ValueError("ActiveZone name must be a non-empty string.")
        if not (isinstance(rect, tuple) or isinstance(rect, list)) or len(rect) != 4:
            raise ValueError("ActiveZone rect must be a tuple or list of 4 numbers.")
        try:
            self.rect = tuple(map(int, rect)) # Ensure coordinates are integers
            if not (self.rect[0] < self.rect[2] and self.rect[1] < self.rect[3]):
                 raise ValueError("Rectangle coordinates must be (x1, y1, x2, y2) with x1 < x2 and y1 < y2.")
        except (ValueError, TypeError) as e:
             raise ValueError(f"Invalid rectangle coordinates: {rect}. {e}")

        self.name = name

    def __repr__(self):
        return f"ActiveZone(name='{self.name}', rect={self.rect})"
# --- End Active Zone class ---


class SyringeVolumeEstimator:
    def __init__(self, active_zones=None, area_threshold=0.9): # Changed parameter name
        """
        Initialize the YOLO model, device, ActiveZones, and other parameters.

        Args:
            active_zones (list[ActiveZone], optional): A list of ActiveZone objects.
                                                       Defaults to None (no active zones).
            area_threshold (float, optional): The minimum proportion of the syringe's
                                              bounding box area that must be inside an AOI
                                              to trigger 'in active zone'. Defaults to 0.9 (90%).
        """
        # Load and evaluate the YOLO model
        self.model = YOLO("runs/pose/train-pose11n-v30/weights/best.pt").eval()
        # Set device based on availability
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        # Define possible syringe diameters in cm
        self.possible_diameters = [0.45, 1.0, 1.25, 2.0]
        # Last timestamp for logging
        self.last_timestamps = deque(maxlen=5)
        # Crop image to center square (1440x1440)
        self.crop_image = False
        # Inference size for YOLO model
        self.inference_size = (1440, 1440)
        # Save video after processing
        self.save_video = False

        # --- Area of Interest (AOI) settings using ActiveZone class ---
        self.active_zones = active_zones if active_zones is not None else [] # List of ActiveZone objects
        # Validate input is a list of ActiveZone objects
        if not isinstance(self.active_zones, list) or not all(isinstance(zone, ActiveZone) for zone in self.active_zones):
             raise TypeError("active_zones must be a list of ActiveZone objects.")
        # Check for duplicate names
        zone_names = [zone.name for zone in self.active_zones]
        if len(zone_names) != len(set(zone_names)):
            print("Warning: Duplicate names found in active_zones. Using the first occurrence.")
            # Optionally, raise an error or handle duplicates differently
            # raise ValueError("ActiveZone names must be unique.")

        self.area_threshold = area_threshold
        # --- End AOI settings ---

    # Removed calculate_iou as it's not needed for the new logic

    def draw_volume_table(self, frame: np.ndarray, volumes: list, table_x: int, table_y: int, track_id: int) -> None:
        """Draw a table on the frame showing diameters and volumes with track ID."""
        table_width = 250
        table_height = 55+len(self.possible_diameters)*30  # For header, track ID, and 4 diameters

        # Create a transparent overlay
        overlay = frame.copy()

        # Draw light gray background on the overlay
        cv2.rectangle(overlay, (table_x, table_y), (table_x + table_width, table_y + table_height), (220, 220, 220), -1)

        # Blend the overlay with the original frame to make it see-through
        alpha = 0.5  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw track ID
        cv2.putText(frame, f"Syringe ID: {track_id}", (table_x + 10, table_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # Draw headers
        cv2.putText(frame, "Diameter", (table_x + 10, table_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "mL", (table_x + 150, table_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # Draw rows for each diameter and volume
        for i, (D, volume) in enumerate(volumes):
            y = table_y + 80 + i * 30
            cv2.putText(frame, f"{D:.2f}", (table_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            if not math.isnan(volume):
                cv2.putText(frame, f"{volume:.2f}", (table_x + 150, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                cv2.putText(frame, "N/A", (table_x + 150, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


    def draw_fps_counter(self, frame: np.ndarray) -> None:
        """Draw a counter on the frame showing the average FPS."""
        current_timestamp = time.time()
        self.last_timestamps.append(current_timestamp)

        if len(self.last_timestamps) >= 2:
            time_diffs = [t2 - t1 for t1, t2 in zip(self.last_timestamps, list(self.last_timestamps)[1:])]
            avg_time = sum(time_diffs) / len(time_diffs)
            avg_fps = 1 / avg_time if avg_time > 0 else 0

            # Add white background rectangle
            text = f"FPS: {avg_fps:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (8, 10), (text_width + 12, 40), (255, 255, 255), -1)

            # Draw text
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


    def process_frame(self, frame: np.ndarray, timestamp: float, writer: csv.writer) -> np.ndarray:
        """
        Process a frame: detect syringes, check ActiveZone overlap (area based),
        calculate volumes, log data (including zone name), and draw annotations.
        """

        original_frame_height, original_frame_width = frame.shape[:2]

        if self.crop_image:
            # Crop the frame to the center square (1440x1440)
            h, w = frame.shape[:2]
            if h > w:
                margin = (h - w) // 2
                frame = frame[margin:margin + w]
            elif w > h:
                margin = (w - h) // 2
                frame = frame[:, margin:margin + h]
            frame = cv2.resize(frame, (1440, 1440))

        # --- Draw Active Zones ---
        display_frame = frame.copy() # Work on a copy for drawing Zones before detection plots
        for zone in self.active_zones:
            x1, y1, x2, y2 = zone.rect
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta color for Zone
            # Add text label with the zone name
            label_y = y1 - 10 if y1 > 20 else y1 + 20 # Adjust label position based on zone y-coordinate
            cv2.putText(display_frame, zone.name, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # --- End Draw Active Zones ---

        # Draw FPS counter on the display frame
        self.draw_fps_counter(display_frame)

        # Perform object detection and tracking
        if self.crop_image:
            results = self.model.track(source=frame,
                                       persist=True,
                                       tracker='bytetrack.yaml',
                                       verbose=False,
                                       conf=0.4,
                                       imgsz=self.inference_size,
                                       )
        else:
             results = self.model.track(source=frame,
                                       persist=True,
                                       tracker='bytetrack.yaml',
                                       verbose=False,
                                       conf=0.5,
                                       )

        # Plot results onto the display_frame (draws default boxes, labels, keypoints etc.)
        annotated_frame = results[0].plot(img=display_frame)

        # Check if there are no results or if there are no detection boxes
        if not results or (hasattr(results[0], "boxes") and len(results[0].boxes) == 0):
            # Log an empty row with NaN values for detection details
            # Add placeholder for zone name
            row = [timestamp, np.nan, np.nan, np.nan] + [np.nan for _ in self.possible_diameters] + ['N/A', 0]
            writer.writerow(row)
            return annotated_frame # Return frame with Zones and FPS drawn

        result = results[0]
        any_syringe_in_active_zone = False # Frame-level flag
        active_zone_status = {} # Store status (True/False) for each box index
        detected_zone_names = {} # Store name of the detected zone for each box index

        # Process each detected syringe (first pass for calculations and logging)
        for i, box in enumerate(result.boxes):
            in_active_zone = False # Syringe-level flag for this specific syringe
            detected_zone_name = "Outside" # Default zone name

            # Extract track ID, assign -1 if not available
            track_id = int(box.id) if box.id is not None else -1

            # Extract bounding box coordinates
            box_coords = box.xyxy[0].cpu().numpy()
            x1_syr, y1_syr, x2_syr, y2_syr = map(int, box_coords)
            syringe_bbox = (x1_syr, y1_syr, x2_syr, y2_syr) # Bbox for calculation

            center_x = (x1_syr + x2_syr) / 2
            center_y = (y1_syr + y2_syr) / 2

            # Calculate syringe bounding box area
            syringe_box_w = x2_syr - x1_syr
            syringe_box_h = y2_syr - y1_syr
            syringe_box_area = float(syringe_box_w * syringe_box_h)

            # --- Check Active Zone overlap (Area based) ---
            if syringe_box_area > 0: # Avoid division by zero if box is invalid
                for zone in self.active_zones:
                    # Get zone rectangle
                    x1_zone, y1_zone, x2_zone, y2_zone = zone.rect

                    # Calculate intersection rectangle
                    inter_x1 = max(syringe_bbox[0], x1_zone)
                    inter_y1 = max(syringe_bbox[1], y1_zone)
                    inter_x2 = min(syringe_bbox[2], x2_zone)
                    inter_y2 = min(syringe_bbox[3], y2_zone)

                    # Calculate intersection area
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    intersection_area = float(inter_w * inter_h)

                    # Check if intersection area is > threshold % of syringe box area
                    if (intersection_area / syringe_box_area) > self.area_threshold:
                        in_active_zone = True
                        any_syringe_in_active_zone = True
                        detected_zone_name = zone.name # Store the name of the zone
                        break # Found the zone for this syringe, no need to check others

            active_zone_status[i] = in_active_zone # Store status by index
            detected_zone_names[i] = detected_zone_name # Store zone name by index
            # --- End Check Active Zone overlap ---

            # Verify keypoints availability
            if result.keypoints is None or len(result.keypoints.xy) <= i or len(result.keypoints.xy[i]) < 4:
                # Log data even if keypoints are missing, indicate zone status and name
                volumes = [np.nan] * len(self.possible_diameters)
                # Use stored status and name for logging
                current_zone_name = detected_zone_names.get(i, "Outside") # Default if somehow missing
                is_in_zone_flag = 1 if active_zone_status.get(i, False) else 0
                row = [timestamp, track_id, center_x, center_y] + volumes + [current_zone_name, is_in_zone_flag]
                writer.writerow(row)
                continue # Skip volume calculation for this syringe

            # Extract keypoints
            try:
                kpts = result.keypoints.xy[i].cpu().numpy()
                ll_point, ul_point, ur_point, lr_point = kpts[:4]  # Lower-left, upper-left, upper-right, lower-right
            except Exception as e:
                print(f"Error extracting keypoints for syringe {track_id}: {e}")
                 # Log data even if keypoint extraction fails, indicate zone status and name
                volumes = [np.nan] * len(self.possible_diameters)
                current_zone_name = detected_zone_names.get(i, "Outside")
                is_in_zone_flag = 1 if active_zone_status.get(i, False) else 0
                row = [timestamp, track_id, center_x, center_y] + volumes + [current_zone_name, is_in_zone_flag]
                writer.writerow(row)
                continue # Skip volume calculation

            # Calculate volumes and log data
            try:
                # Calculate width and height in pixels (average of top/bottom and left/right)
                width_pixels = (np.linalg.norm(lr_point - ll_point) + np.linalg.norm(ur_point - ul_point)) / 2
                height_pixels = (np.linalg.norm(ul_point - ll_point) + np.linalg.norm(ur_point - lr_point)) / 2
                if width_pixels <= 0 or height_pixels <= 0:
                     # Log data even if dimensions are invalid, indicate zone status and name
                    volumes = [np.nan] * len(self.possible_diameters)
                    current_zone_name = detected_zone_names.get(i, "Outside")
                    is_in_zone_flag = 1 if active_zone_status.get(i, False) else 0
                    row = [timestamp, track_id, center_x, center_y] + volumes + [current_zone_name, is_in_zone_flag]
                    writer.writerow(row)
                    continue

                # Calculate volumes for all possible diameters
                volumes = []
                for D in self.possible_diameters:
                    scale_factor_D = D / width_pixels
                    H_cm = height_pixels * scale_factor_D
                    if 0 < H_cm <= 30:  # Validate height (max 30 cm)
                        volume_D = math.pi * (D / 2) ** 2 * H_cm
                    else:
                        volume_D = float('nan')
                    volumes.append(volume_D)

                # Log data to CSV (including active zone status and name)
                current_zone_name = detected_zone_names.get(i, "Outside")
                is_in_zone_flag = 1 if active_zone_status.get(i, False) else 0
                row = [timestamp, track_id, center_x, center_y] + volumes + [current_zone_name, is_in_zone_flag]
                writer.writerow(row)

                # Draw volume table
                table_x = x2_syr + 10  # Right of bounding box
                table_y = y1_syr       # Top of bounding box
                self.draw_volume_table(annotated_frame, list(zip(self.possible_diameters, volumes)), table_x, table_y, track_id)

            except Exception as e:
                print(f"Error processing syringe {track_id}: {e}")
                # Log data even if volume calculation fails, indicate zone status and name
                volumes = [np.nan] * len(self.possible_diameters)
                current_zone_name = detected_zone_names.get(i, "Outside")
                is_in_zone_flag = 1 if active_zone_status.get(i, False) else 0
                row = [timestamp, track_id, center_x, center_y] + volumes + [current_zone_name, is_in_zone_flag]
                writer.writerow(row)
                continue

        # --- Second Pass: Redraw active boxes in green ---
        # This loop goes *after* the main processing loop and *after* results[0].plot()
        if hasattr(result, "boxes"): # Ensure boxes exist
            for i, box in enumerate(result.boxes):
                # Check the stored status for this box index
                if active_zone_status.get(i, False):
                    # Get coordinates again
                    box_coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box_coords)
                    # Draw a green rectangle over the one drawn by plot()
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green color, thickness 2
                    # Optionally, add the zone name to the box label (might get cluttered)
                    # zone_name_label = detected_zone_names.get(i, "")
                    # label_text = f"ID:{int(box.id)} {zone_name_label}" # Example combining ID and zone
                    # cv2.putText(annotated_frame, zone_name_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- End Redraw ---


        # --- Draw overall Active Zone indicator ---
        if any_syringe_in_active_zone:
            text = "ACTIVE ZONE DETECTED"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            # Position at bottom center
            text_x = (annotated_frame.shape[1] - text_width) // 2
            text_y = annotated_frame.shape[0] - 20
             # Draw white background rectangle
            cv2.rectangle(annotated_frame, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (255, 255, 255), -1)
            # Draw green text
            cv2.putText(annotated_frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 180, 0), 2) # Slightly darker green text
        # --- End Active Zone indicator ---

        return annotated_frame

    def run(self, input_source='webcam', video_path=None, csv_path='syringe_data.csv'):
        """Run the main loop to process frames from webcam or video, saving data and optionally video."""

        # Delete existing CSV file if it exists
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                print(f"Removed existing CSV file: {csv_path}")
            except OSError as e:
                print(f"Error removing file {csv_path}: {e}. Appending data instead.")


        # Set up video capture based on input source
        if input_source == 'video':
            if video_path is None:
                raise ValueError("video_path must be provided for input_source='video'")
            cap = cv2.VideoCapture(video_path)
            # Set up video writer for video input
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"🎥 Video resolution: {width} x {height}, FPS: {fps}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_processed{ext}"

            if self.save_video:
                # Ensure writer uses correct dimensions
                write_width, write_height = (1440, 1440) if self.crop_image else (width, height)
                out = cv2.VideoWriter(output_path, fourcc, fps, (write_width, write_height))
                print(f"💾 Saving processed video to: {output_path} with resolution {write_width}x{write_height}")
            else:
                out = None

        else:  # webcam
            cap = cv2.VideoCapture(0)
            # Set camera resolution attempt (actual resolution might differ)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            # Verify camera settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"🎥 Actual webcam resolution: {actual_width} x {actual_height}")
            # Use actual resolution if cropping is not enabled
            width, height = actual_width, actual_height
            out = None # No video saving for webcam by default

        if not cap.isOpened():
            raise IOError(f"Cannot open {'video file' if input_source == 'video' else 'webcam'}")

        # Open CSV file for logging
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header only if file is new/empty
            if not file_exists:
                # --- Updated Header ---
                header = ['timestamp', 'track_id', 'center_x', 'center_y'] + \
                         [f'volume_D{D}' for D in self.possible_diameters] + \
                         ['zone_name', 'in_active_zone'] # Added zone_name
                # --- End Updated Header ---
                writer.writerow(header)

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video or camera stream.")
                        break
                    # Determine timestamp
                    if input_source == 'video':
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Video time in seconds
                    else:
                        timestamp = time.time()  # System time for webcam
                    # Process frame
                    annotated_frame = self.process_frame(frame, timestamp, writer)
                    # Write to output video if processing a video file and saving is enabled
                    if out is not None and self.save_video:
                         # Ensure the frame size matches the writer dimensions before writing
                        if annotated_frame.shape[1] != write_width or annotated_frame.shape[0] != write_height:
                            annotated_frame_resized = cv2.resize(annotated_frame, (write_width, write_height))
                            out.write(annotated_frame_resized)
                        else:
                             out.write(annotated_frame)

                    # Display frame
                    # Resize for display if it's too large (e.g., 4K on a smaller monitor)
                    display_scale = 0.5 # Scale down 4K to roughly 1080p width for display
                    if annotated_frame.shape[1] > 1920:
                         display_height = int(annotated_frame.shape[0] * display_scale)
                         display_width = int(annotated_frame.shape[1] * display_scale)
                         annotated_frame_display = cv2.resize(annotated_frame, (display_width, display_height))
                    else:
                         annotated_frame_display = annotated_frame

                    cv2.imshow('Syringe Volume Measurement', annotated_frame_display)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quitting...")
                        break
            finally:
                print("Releasing resources...")
                cap.release()
                if out is not None and self.save_video:
                    out.release()
                    print("Output video saved.")
                cv2.destroyAllWindows()
                print("Windows closed.")


if __name__ == "__main__":
    # --- Define your Areas of Interest using the ActiveZone class ---
    # Use coordinates relative to the frame resolution being processed by YOLO
    # (e.g., original webcam resolution if crop_image=False, or 1440x1440 if crop_image=True)

    # Example for a 1920x1080 frame (adjust coordinates as needed):
    try:
        zone_list = [
            ActiveZone(name="Abdomen", rect=(100, 100, 500, 980)),    # A tall rectangle on the left
            ActiveZone(name="Head", rect=(1400, 600, 1800, 1000)) # Bottom-right area
        ]
    except ValueError as e:
        print(f"Error creating ActiveZone: {e}")
        zone_list = [] # Fallback to empty list if definition is wrong


    # Example for a 3840x2160 (4K) frame:
    # try:
    #     zone_list = [
    #         ActiveZone(name="Microscope Zone", rect=(500, 500, 1500, 1500)),    # An area somewhere in the upper-left region
    #         ActiveZone(name="Waste Chute", rect=(2500, 1000, 3500, 2000))   # An area somewhere in the lower-right region
    #     ]
    # except ValueError as e:
    #     print(f"Error creating ActiveZone: {e}")
    #     zone_list = []

    # If you don't want any AOIs, use: zone_list = []
    # -------------------------------------------

    # Instantiate the estimator with the defined ActiveZones and area threshold (e.g. 90%)
    try:
        estimator = SyringeVolumeEstimator(active_zones=zone_list, area_threshold=0.9) # Pass list of objects

        # --- Choose how to run ---
        # Option 1: Webcam
        estimator.run(input_source='webcam', csv_path='webcam_syringe_data.csv') # New CSV name

        # Option 2: Video File
        # estimator.save_video = True # Set to True if you want to save the processed video
        # estimator.run(input_source='video', video_path='your_video.mov', csv_path='video_syringe_data_zones.csv') # New CSV name
        # -------------------------

    except (TypeError, ValueError) as e:
         print(f"Error initializing SyringeVolumeEstimator: {e}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
