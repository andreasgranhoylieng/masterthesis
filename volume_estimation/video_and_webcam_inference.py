import csv
import math
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class SyringeVolumeEstimator:
    def __init__(self):
        """Initialize the YOLO model, device, and possible diameters."""
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

        """Process a frame to detect syringes, calculate volumes, log data, and draw tables."""


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

        # Draw FPS counter
        self.draw_fps_counter(frame)

        # Define the desired inference image size (matching your training size)


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


        # Check if there are no results or if there are no detection boxes
        if not results or (hasattr(results[0], "boxes") and len(results[0].boxes) == 0):
            # Log an empty row with NaN values for detection details
            row = [timestamp, np.nan, np.nan, np.nan] + [np.nan for _ in self.possible_diameters]
            writer.writerow(row)
            return frame

        result = results[0]
        annotated_frame = result.plot()  # Draw bounding boxes with track IDs

        # Process each detected syringe
        for i, box in enumerate(result.boxes):
            # Extract track ID, assign -1 if not available
            track_id = int(box.id) if box.id is not None else -1

            # Extract bounding box coordinates
            box_coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box_coords)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Verify keypoints availability
            if result.keypoints is None or len(result.keypoints.xy) <= i or len(result.keypoints.xy[i]) < 4:
                continue

            # Extract keypoints
            try:
                kpts = result.keypoints.xy[i].cpu().numpy()
                ll_point, ul_point, ur_point, lr_point = kpts[:4]  # Lower-left, upper-left, upper-right, lower-right
            except Exception as e:
                print(f"Error extracting keypoints for syringe {track_id}: {e}")
                continue

            # Calculate volumes and log data
            try:
                # Calculate width and height in pixels (average of top/bottom and left/right)
                width_pixels = (np.linalg.norm(lr_point - ll_point) + np.linalg.norm(ur_point - ul_point)) / 2
                height_pixels = (np.linalg.norm(ul_point - ll_point) + np.linalg.norm(ur_point - lr_point)) / 2
                if width_pixels <= 0 or height_pixels <= 0:
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

                # Log data to CSV
                row = [timestamp, track_id, center_x, center_y] + volumes
                writer.writerow(row)

                # Draw volume table
                table_x = x2 + 10  # Right of bounding box
                table_y = y1       # Top of bounding box
                self.draw_volume_table(annotated_frame, list(zip(self.possible_diameters, volumes)), table_x, table_y, track_id)

            except Exception as e:
                print(f"Error processing syringe {track_id}: {e}")
                continue



        return annotated_frame

    def run(self, input_source='webcam', video_path=None, csv_path='syringe_data.csv'):
        """Run the main loop to process frames from webcam or video, saving data and optionally video."""

        # Delete existing CSV file if it exists
        if os.path.exists(csv_path):
            os.remove(csv_path)


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
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            else:
                out = None

        else:  # webcam
            cap = cv2.VideoCapture(0)
            # Set camera resolution to 4K
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            # Verify camera settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"🎥 Actual webcam resolution: {actual_width} x {actual_height}")
            out = None

        if not cap.isOpened():
            raise IOError(f"Cannot open {'video file' if input_source == 'video' else 'webcam'}")

        # Open CSV file for logging
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header if file is new or empty
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                header = ['timestamp', 'track_id', 'center_x', 'center_y'] + [f'volume_D{D}' for D in self.possible_diameters]
                writer.writerow(header)

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Determine timestamp
                    if input_source == 'video':
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Video time in seconds
                    else:
                        timestamp = time.time()  # System time for webcam
                    # Process frame
                    annotated_frame = self.process_frame(frame, timestamp, writer)
                    # Write to output video if processing a video file
                    if out is not None:
                        out.write(annotated_frame)
                    # Display frame
                    cv2.imshow('Syringe Volume Measurement', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cap.release()
                if out is not None:
                    out.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    estimator = SyringeVolumeEstimator()
    # Example usage for webcam
    # estimator.run(input_source='webcam')
    # Example usage for video
    # estimator.run(input_source='video', video_path='IMG_4952.mov')
