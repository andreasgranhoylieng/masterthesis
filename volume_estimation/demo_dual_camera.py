# --- Syringe Workflow Detection Script (Dual Webcam Mode - Simplified Logic v2 - No Timeouts, Persistent Vol Check) ---
# Detects syringes, estimates volume, tracks them,
# and validates a workflow based on simple disappearance/appearance events.
# Volume is checked PERSISTENTLY after insertion until found.
# Program attempts to exit automatically after a successful return cycle.
# WARNING: This simplified logic is sensitive to occlusions and detection flickers.

import csv
import math
import os
import time
from collections import deque, defaultdict
import sys
import traceback
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Configuration Flag for Debug Logging ---
# Set to True to enable detailed print statements in process_frame
DEBUG_MODE = False
# --------------------------------------------

# --- Define the Active Zone class ---
class ActiveZone:
    """Represents a named rectangular area of interest."""

    def __init__(self, name: str, rect: tuple):
        if not isinstance(name, str) or not name:
            raise ValueError("ActiveZone name must be a non-empty string.")
        if not (isinstance(rect, tuple) or isinstance(rect, list)) or len(rect) != 4:
            raise ValueError("ActiveZone rect must be a tuple or list of 4 numbers.")
        try:
            self.rect = tuple(map(int, rect))
            if not (self.rect[0] < self.rect[2] and self.rect[1] < self.rect[3]):
                raise ValueError(
                    "Rectangle coordinates must be (x1, y1, x2, y2) with x1 < x2 and"
                    " y1 < y2."
                )
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

    def __init__(
        self,
        model_path: str,
        possible_diameters_cm: list[float],
        active_zones: list[ActiveZone] = None,
        area_threshold: float = 0.8,
        device_preference: str = None,
    ):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. Please check the path."
            )
        try:
            self.model = YOLO(self.model_path).eval()
        except Exception as e:
            print(f"Error loading YOLO model from {self.model_path}: {e}")
            print(
                "Please ensure the model path is correct, the file exists, and"
                " dependencies are installed."
            )
            raise

        self.device = None
        if device_preference and device_preference in ["cuda", "mps", "cpu"]:
            self.device = device_preference
            if self.device == "cuda" and not torch.cuda.is_available():
                print(f"Warning: Preferred device 'cuda' not available. Falling back.")
                self.device = None
            elif self.device == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                print(f"Warning: Preferred device 'mps' not available. Falling back.")
                self.device = None
        if self.device is None:
            if torch.cuda.is_available(): self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): self.device = "mps"
            else: self.device = "cpu"

        print(f"Estimator using device: {self.device}")
        self.model.to(self.device)

        if not isinstance(possible_diameters_cm, list) or not all(
            isinstance(d, (float, int)) and d > 0 for d in possible_diameters_cm
        ):
            raise ValueError("possible_diameters_cm must be a list of positive numbers.")
        self.possible_diameters = sorted(possible_diameters_cm)

        self.active_zones = active_zones if active_zones is not None else []
        if not isinstance(self.active_zones, list) or not all(
            isinstance(zone, ActiveZone) for zone in self.active_zones
        ):
            raise TypeError("active_zones must be a list of ActiveZone objects.")
        zone_names = [zone.name for zone in self.active_zones]
        if len(zone_names) != len(set(zone_names)):
            raise ValueError(f"ActiveZone names must be unique: {zone_names}")
        self.area_threshold = max(0.1, min(1.0, area_threshold))
        print(f"  Zones configured: {[zone.name for zone in self.active_zones]}")

        self.last_timestamps = deque(maxlen=10)
        self.save_video = False

    def draw_volume_table( self, frame: np.ndarray, volumes: list[tuple[float, float]], table_x: int, table_y: int, track_id: int) -> None:
        frame_h, frame_w = frame.shape[:2]; table_width = 250; row_height = 25; header_height = 50
        table_height = header_height + len(volumes) * row_height + 5
        if table_x + table_width > frame_w - 10: table_x = frame_w - table_width - 10
        if table_y + table_height > frame_h - 10: table_y = frame_h - table_height - 10
        if table_x < 10: table_x = 10
        if table_y < 10: table_y = 10
        table_x, table_y = int(table_x), int(table_y)
        try:
            overlay = frame.copy(); cv2.rectangle(overlay, (table_x, table_y), (table_x + table_width, table_y + table_height), (210, 210, 210), -1); alpha = 0.6; cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        except Exception as e: print(f"Warning: Error drawing table background: {e}")
        text_color=(0,0,0); font=cv2.FONT_HERSHEY_SIMPLEX; font_scale=0.5; thickness=1
        cv2.putText(frame, f"Syringe ID: {track_id}", (table_x + 10, table_y + 20), font, 0.6, text_color, thickness + 1)
        cv2.putText(frame, "Diam (cm)", (table_x + 10, table_y + header_height - 10), font, font_scale, text_color, thickness)
        cv2.putText(frame, "Volume (mL)", (table_x + 120, table_y + header_height - 10), font, font_scale, text_color, thickness)
        for i, (diameter, volume) in enumerate(volumes):
            row_y = table_y + header_height + i * row_height + (row_height // 2)
            cv2.putText(frame, f"{diameter:.2f}", (table_x + 10, row_y), font, font_scale, text_color, thickness)
            display_vol = f"{volume:.2f}" if volume is not None and not math.isnan(volume) and volume >= 0 else "N/A"
            cv2.putText(frame, display_vol, (table_x + 120, row_y), font, font_scale, text_color, thickness)

    def draw_fps_counter(self, frame: np.ndarray, prefix: str = "FPS") -> None:
        current_timestamp = time.monotonic(); self.last_timestamps.append(current_timestamp)
        if len(self.last_timestamps) > 1:
            try:
                time_diffs = np.diff(list(self.last_timestamps)); valid_diffs = time_diffs[time_diffs < 1.0]
                avg_fps = (1.0 / np.mean(valid_diffs)) if len(valid_diffs) > 0 else 0
                text = f"{prefix}: {avg_fps:.1f}"; font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.7; thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                rect_x1, rect_y1 = 8, 10; rect_x2, rect_y2 = rect_x1 + text_width + 12, rect_y1 + text_height + baseline + 5
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
                cv2.putText(frame, text, (rect_x1 + 6, rect_y2 - baseline), font, font_scale, (0, 0, 0), thickness)
            except Exception as e: print(f"Warning: Error calculating FPS: {e}")

    def process_frame( self, frame: np.ndarray, timestamp: float, writer: csv.writer = None, camera_source: str = "unknown") -> tuple[np.ndarray, list]:
        original_frame_height, original_frame_width = frame.shape[:2]; annotated_frame = frame.copy()
        for zone in self.active_zones:
            x1,y1,x2,y2=zone.rect; x1,y1=max(0,x1),max(0,y1); x2,y2=min(original_frame_width-1,x2),min(original_frame_height-1,y2)
            if x1>=x2 or y1>=y2: continue
            cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(255,0,255),2)
            label_y = y1-10 if y1>30 else y1+20; label_y=max(15,min(label_y,original_frame_height-5))
            cv2.putText(annotated_frame,zone.name,(x1+5,label_y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        fps_prefix=f"{camera_source.capitalize()} FPS"; self.draw_fps_counter(annotated_frame, prefix=fps_prefix)
        processed_detections=[]; results_data=None
        try:
            results=self.model.track(source=frame,persist=True,tracker="bytetrack.yaml",verbose=False,conf=0.3,device=self.device,classes=[0])
            if results and len(results) > 0: results_data=results[0].cpu(); annotated_frame=results_data.plot(img=annotated_frame, line_width=1, font_size=0.4)
            else: return annotated_frame, []
        except Exception as e:
            print(f"Error during YOLO tracking/plotting ({camera_source}): {e}")
            if writer: row=[timestamp, camera_source, np.nan, np.nan, np.nan]+[np.nan for _ in self.possible_diameters]+["TrackingError", 0]; writer.writerow(row)
            return annotated_frame, []
        if results_data is None or results_data.boxes is None or results_data.boxes.id is None: return annotated_frame, []

        track_id = None # Initialize track_id before the loop
        for i, box in enumerate(results_data.boxes):
            if box.id is None:
                if DEBUG_MODE: print(f"DEBUG: Skipping box {i} because track ID is None.")
                continue
            track_id = int(box.id[0])
            box_coords=box.xyxy[0].numpy(); x1_syr,y1_syr,x2_syr,y2_syr=map(int,box_coords)
            if x1_syr>=x2_syr or y1_syr>=y2_syr:
                if DEBUG_MODE: print(f"DEBUG: Skipping box {i}, ID {track_id} due to invalid bbox.")
                continue
            syringe_bbox=(x1_syr,y1_syr,x2_syr,y2_syr); center_x=float((x1_syr+x2_syr)/2); center_y=float((y1_syr+y2_syr)/2); syringe_box_area=float((x2_syr-x1_syr)*(y2_syr-y1_syr))
            in_active_zone_flag=False; detected_zone_name="Outside"; max_overlap_ratio=0.0
            if syringe_box_area>0:
                for zone in self.active_zones:
                    x1_zone,y1_zone,x2_zone,y2_zone=zone.rect; inter_x1=max(x1_syr,x1_zone); inter_y1=max(y1_syr,y1_zone); inter_x2=min(x2_syr,x2_zone); inter_y2=min(y2_syr,y2_zone)
                    inter_w=max(0,inter_x2-inter_x1); inter_h=max(0,inter_y2-inter_y1); intersection_area=float(inter_w*inter_h); overlap_ratio=intersection_area/syringe_box_area if syringe_box_area>0 else 0
                    if overlap_ratio>=self.area_threshold and overlap_ratio>max_overlap_ratio: max_overlap_ratio=overlap_ratio; detected_zone_name=zone.name; in_active_zone_flag=True
            volumes_for_diameters={D: float("nan") for D in self.possible_diameters}
            width_px, height_px = -1.0, -1.0 # Init values for debugging
            kpts_conf = None

            if (results_data.keypoints is not None and len(results_data.keypoints.xy)>i and results_data.keypoints.conf is not None and len(results_data.keypoints.conf)>i and results_data.keypoints.xy[i].shape[0]>=4):
                try:
                    kpts=results_data.keypoints.xy[i][:4].numpy()
                    kpts_conf = results_data.keypoints.conf[i][:4].numpy() # Get confidences too
                    ll_point,ul_point,ur_point,lr_point=kpts
                    width_px=(np.linalg.norm(lr_point-ll_point)+np.linalg.norm(ur_point-ul_point))/2.0
                    height_px=(np.linalg.norm(ul_point-ll_point)+np.linalg.norm(ur_point-lr_point))/2.0

                    # --- DEBUG LOGGING START ---
                    if DEBUG_MODE and camera_source == 'manikin':
                        kpts_conf_str = ", ".join([f"{c:.2f}" for c in kpts_conf])
                        print(f"DEBUG (Manikin Cam) ID: {track_id}, Kpts Conf: [{kpts_conf_str}], width_px: {width_px:.2f}, height_px: {height_px:.2f}")
                    # --- DEBUG LOGGING END ---

                    if width_px > 2 and height_px > 2: # Check pixel dimensions
                        for D_cm in self.possible_diameters:
                            H_cm = float('nan') # Init H_cm for debug logging
                            if width_px<=0: continue # Safety check
                            scale_factor_cm_per_px=D_cm/width_px
                            H_cm=height_px*scale_factor_cm_per_px
                            # --- DEBUG LOGGING START ---
                            if DEBUG_MODE and camera_source == 'manikin': # and D_cm == 1.0: # Optionally focus on correct diameter
                                 print(f"DEBUG (Manikin Cam) ID: {track_id}, Diam: {D_cm:.2f}, H_cm: {H_cm:.2f}")
                            # --- DEBUG LOGGING END ---
                            if 0<H_cm<=30.0: # Use 30cm limit like old script (adjust if 25cm is truly intended)
                                radius_cm=D_cm/2.0
                                volume_ml=math.pi*(radius_cm**2)*H_cm
                                volumes_for_diameters[D_cm]=volume_ml
                            # else: H_cm is invalid, volume remains NaN
                    # else: width_px or height_px too small, all volumes remain NaN

                except Exception as e:
                    if DEBUG_MODE: print(f"DEBUG: Exception during volume calculation for ID {track_id}: {e}")
                    pass # Keep volumes as NaN
            else:
                 # --- DEBUG LOGGING START ---
                 if DEBUG_MODE and camera_source == 'manikin':
                      kp_info = "None"
                      if results_data.keypoints is not None:
                           if len(results_data.keypoints.xy) <= i: kp_info = f"Index {i} out of bounds (len {len(results_data.keypoints.xy)})"
                           elif results_data.keypoints.xy[i].shape[0] < 4: kp_info = f"Only {results_data.keypoints.xy[i].shape[0]} points found"
                           elif len(results_data.keypoints.conf) <= i: kp_info = f"Conf index {i} out of bounds"
                      print(f"DEBUG (Manikin Cam) ID: {track_id}, Keypoints insufficient/missing ({kp_info})")
                 # --- DEBUG LOGGING END ---
                 pass # Keep volumes as NaN if keypoints inadequate


            if track_id is None: # Should not happen if checks above work, but safety
                print("Warning: track_id became None unexpectedly before creating detection_info.")
                continue
            detection_info={"id":track_id, "zone":detected_zone_name, "volumes":volumes_for_diameters, "bbox":syringe_bbox, "center":(center_x,center_y), "in_active_zone_flag":in_active_zone_flag, "camera_source":camera_source}
            processed_detections.append(detection_info)
            if writer:
                volumes_list=[volumes_for_diameters.get(D,np.nan) for D in self.possible_diameters]; log_row=[timestamp,camera_source,track_id,center_x,center_y]+volumes_list+[detected_zone_name, 1 if in_active_zone_flag else 0]
                try: writer.writerow(log_row)
                except Exception as e: print(f"Error writing row to CSV ({camera_source}): {e}")
            volumes_for_table=list(volumes_for_diameters.items()); table_x=x2_syr+10; table_y=y1_syr; self.draw_volume_table(annotated_frame,volumes_for_table,table_x,table_y,track_id)
        # --- End of loop for boxes ---

        for det in processed_detections:
            if det["in_active_zone_flag"]: x1,y1,x2,y2=map(int,det["bbox"]); cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(0,255,0),2)
        if any(d["in_active_zone_flag"] for d in processed_detections):
            text=f"ACTIVE ZONE DETECTED ({camera_source})"; font=cv2.FONT_HERSHEY_SIMPLEX; font_scale=0.8; thickness=2; (text_width,text_height),baseline=cv2.getTextSize(text,font,font_scale,thickness)
            text_x=(annotated_frame.shape[1]-text_width)//2; text_y=annotated_frame.shape[0]-15; rect_y1=text_y-text_height-baseline-5; rect_y2=annotated_frame.shape[0]-10
            cv2.rectangle(annotated_frame,(text_x-10,rect_y1),(text_x+text_width+10,rect_y2),(255,255,255),-1); cv2.putText(annotated_frame,text,(text_x,text_y-baseline//2),font,font_scale,(0,180,0),thickness)
        return annotated_frame, processed_detections

    def setup_video_writer( self, output_path: str, frame_width: int, frame_height: int, fps: float) -> cv2.VideoWriter:
        if not self.save_video: return None
        if fps <= 0: fps = 25.0; print(f"Warning: Invalid FPS provided ({fps}). Using default: {fps:.1f} FPS")
        if frame_width <= 0 or frame_height <= 0: print(f"Error: Invalid frame dimensions ({frame_width}x{frame_height}). Cannot create video writer."); self.save_video = False; return None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v"); os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened(): raise IOError(f"VideoWriter failed to open for path: {output_path}")
            print(f"💾 Saving processed video to: {output_path} ({frame_width}x{frame_height} @ {fps:.1f} FPS)")
            return out
        except Exception as e: print(f"Error initializing VideoWriter for {output_path}: {e}. Video saving disabled."); self.save_video = False; return None
# --- End of SyringeVolumeEstimator Class ---


# --- Define the Syringe Test Workflow Class (Simplified Logic v2 - No Timeouts, Persistent Vol Check) ---
class SyringeTestWorkflow:
    """
    Manages and validates a workflow using simplified disappearance/appearance logic.
    Volume checked persistently after insertion. No timeouts. Auto-exits on successful return.
    WARNING: Sensitive to occlusions and detection flickers.
    """
    STATE_IDLE = "IDLE"
    STATE_SYRINGE_PICKED = "SYRINGE_PICKED"
    STATE_SYRINGE_INSERTED = "SYRINGE_INSERTED"

    ERROR_WRONG_SYRINGE = "Wrong Syringe Picked"
    ERROR_WRONG_VOLUME = "Incorrect Volume" # Logged once volume is determined
    ERROR_WRONG_TARGET = "Wrong Target Zone"
    ERROR_MULTI_ACTIVE = "Multiple Syringes Missing"
    ERROR_LOST_TRACK = "Lost Track of Syringe" # Less likely without timeouts, but kept for state logic
    ERROR_PREMATURE_RETURN = "Syringe Returned Prematurely"
    # ERROR_VOLUME_UNDETERMINED = "Volume Undetermined" # Replaced by persistent check
    ERROR_UNEXPECTED_INSERT = "Unexpected Syringe Insertion"
    ERROR_UNEXPECTED_RETURN = "Unexpected Syringe Return"

    def __init__(
        self,
        table_zone_names: list[str],
        target_zone_names: list[str],
        correct_starting_zone: str,
        correct_syringe_diameter: float,
        possible_diameters: list[float],
        target_volume_ml: float,
        volume_tolerance_ml: float,
        correct_target_zone: str,
        log_file_path: str = "workflow_log.txt",
        # Timeouts removed
    ):
        self.table_zones = set(table_zone_names)
        self.target_zones = set(target_zone_names)
        self.correct_starting_zone = correct_starting_zone
        self.correct_diameter = correct_syringe_diameter
        self.possible_diameters = possible_diameters
        self.target_volume = target_volume_ml
        self.volume_tolerance = volume_tolerance_ml
        self.correct_target_zone = correct_target_zone

        self.current_state = self.STATE_IDLE
        self.active_syringe_id = None
        self.active_id_source = None
        self.active_syringe_start_zone = None
        self.active_syringe_current_zone = None
        self.active_syringe_volume = None # Stores the first valid volume found
        self.volume_determined_this_cycle = False # Flag for persistent check
        self.error_flags_this_cycle = set()
        self.last_state_update_time = time.monotonic() # Still useful for state duration if needed later

        self.syringe_last_known_zone = defaultdict(lambda: "Unknown") # Key: (track_id, source)
        self.syringe_last_seen_time = defaultdict(float) # Key: (track_id, source)
        # Removed global_last_seen and purge timeout

        self.syringes_on_table = {} # Stores {syringes_cam_track_id: zone_name}
        self.initial_table_scan_complete = False
        self.workflow_completed = False # Flag to signal successful completion

        self.log_file_path = log_file_path
        self._log_entries = []
        self._clear_log_file()
        self._log_header()

    def _clear_log_file(self):
        try:
            os.makedirs(os.path.dirname(self.log_file_path) or ".", exist_ok=True)
            with open(self.log_file_path, "w") as f: f.write("")
        except IOError as e: print(f"Warning: Could not clear log file {self.log_file_path}: {e}")

    def _log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} [{level:<5}] [{self.current_state}] {message}"
        print(log_message)
        self._log_entries.append(log_message)
        try:
            with open(self.log_file_path, "a") as f: f.write(log_message + "\n")
        except IOError as e: print(f"Warning: Could not write to log file {self.log_file_path}: {e}")

    def _log_error(self, error_type: str, message: str):
        self._log(f"{error_type}: {message}", level="ERROR")
        self.error_flags_this_cycle.add(error_type)

    def _log_header(self):
        self._log("--- Workflow Initialized (Simplified Logic v2 - No Timeout / Persistent Vol) ---", level="SETUP")
        self._log(f"Table Zones (Syringe Cam): {list(self.table_zones)}", level="SETUP")
        self._log(f"Target Zones (Manikin Cam): {list(self.target_zones)}", level="SETUP")
        self._log(f"Correct Start Zone: '{self.correct_starting_zone}'", level="SETUP")
        self._log(f"Correct Diameter: {self.correct_diameter} cm", level="SETUP")
        self._log(f"Target Volume: {self.target_volume:.2f} +/- {self.volume_tolerance:.2f} mL",level="SETUP")
        self._log(f"Correct Target Zone: '{self.correct_target_zone}'", level="SETUP")
        self._log("--- Waiting for Action ---", level="SETUP")

    def _reset_state(self, reason: str):
        """Resets workflow state and logs summary if ending a cycle."""
        if self.current_state != self.STATE_IDLE:
            if self.active_syringe_id is not None:
                id_to_report = self.active_syringe_id; source_cam = self.active_id_source
                summary = f"Cycle ended for syringe ID {id_to_report} (from {source_cam} cam)."
                if self.error_flags_this_cycle: summary += f" Errors: {', '.join(sorted(list(self.error_flags_this_cycle)))}."
                elif self.current_state == self.STATE_SYRINGE_INSERTED and not self.workflow_completed:
                     summary += " Status: Returned OK (but prior errors existed or completion flag not set)."
                elif self.workflow_completed:
                     summary += " Status: OK." # Logged separately before reset now
                else: summary += " Status: Aborted."
                self._log(summary, level="SUMMARY")
            self._log(f"Resetting state to IDLE. Reason: {reason}")

        self.current_state = self.STATE_IDLE
        self.active_syringe_id = None; self.active_id_source = None
        self.active_syringe_start_zone = None; self.active_syringe_current_zone = None
        self.active_syringe_volume = None; self.error_flags_this_cycle = set()
        self.volume_determined_this_cycle = False # Reset volume flag
        self.last_state_update_time = time.monotonic()
        # self.workflow_completed = False # Reset completion flag for next potential run

    def _update_tracking_history(self, all_detections, current_time):
        """Updates last known zone and time for each detection."""
        for det in all_detections:
            source = det.get('camera_source', 'unknown')
            track_id = det.get('id')
            zone = det.get('zone')
            if track_id is not None and track_id != -1 and source != 'unknown':
                key = (track_id, source)
                self.syringe_last_known_zone[key] = zone
                self.syringe_last_seen_time[key] = current_time
                # Removed global_last_seen update

    # Removed _purge_old_tracks method

    def _initialize_table_state(self, syringe_detections):
        """Populate the initial state of syringes on the table."""
        if self.initial_table_scan_complete: return
        if DEBUG_MODE: print("DEBUG: Performing initial table scan...")
        temp_table_state = {}
        for det in syringe_detections:
            if det['zone'] in self.table_zones:
                temp_table_state[det['id']] = det['zone']
                if DEBUG_MODE: print(f"DEBUG: Initial scan - Found ID {det['id']} in {det['zone']}")
        # Require at least one fewer syringe than zones, or just one if only 1 zone
        required_count = max(1, len(self.table_zones) -1)
        if len(temp_table_state) >= required_count :
             self.syringes_on_table = temp_table_state
             self.initial_table_scan_complete = True
             self._log("Initial table state identified.", level="INFO")
             if DEBUG_MODE: print(f"DEBUG: Initial table state set: {self.syringes_on_table}")
        elif DEBUG_MODE: print(f"DEBUG: Initial scan found only {len(temp_table_state)}. Waiting...")

    def update_state(self, all_detections: list[dict], current_time: float):
        """Updates workflow state using simplified logic. No timeouts. Persistent volume check."""
        self._update_tracking_history(all_detections, current_time)
        # Removed purge call

        detections_by_source = {'manikin': [], 'syringes': []}
        current_syringe_cam_detections = {}
        for det in all_detections:
            source = det.get('camera_source'); track_id = det.get('id')
            if source in detections_by_source and track_id is not None and track_id != -1:
                detections_by_source[source].append(det)
                if source == 'syringes': current_syringe_cam_detections[track_id] = det

        if not self.initial_table_scan_complete and self.current_state == self.STATE_IDLE:
            self._initialize_table_state(detections_by_source['syringes'])
            if not self.initial_table_scan_complete: return

        # --- Simplified State Machine ---

        if self.current_state == self.STATE_IDLE:
            if not self.initial_table_scan_complete: return

            current_ids_on_table = {det['id'] for det in detections_by_source['syringes'] if det['zone'] in self.table_zones}
            expected_ids_on_table = set(self.syringes_on_table.keys())
            missing_ids = expected_ids_on_table - current_ids_on_table

            if len(missing_ids) == 1:
                pickup_id = list(missing_ids)[0]
                origin_zone = self.syringes_on_table.get(pickup_id, "Unknown Zone")
                self._log(f"Syringe ID {pickup_id} (syringes cam) disappeared from '{origin_zone}'. State -> {self.STATE_SYRINGE_PICKED}.")

                self.active_syringe_id = pickup_id
                self.active_id_source = 'syringes'
                self.active_syringe_start_zone = origin_zone
                self.active_syringe_current_zone = "Presumed Outside"
                self.current_state = self.STATE_SYRINGE_PICKED
                self.error_flags_this_cycle = set()
                self.volume_determined_this_cycle = False # Ensure reset here too
                self.last_state_update_time = current_time
                if pickup_id in self.syringes_on_table: del self.syringes_on_table[pickup_id]

                if origin_zone != self.correct_starting_zone:
                    self._log_error(self.ERROR_WRONG_SYRINGE, f"Picked up ID {pickup_id} from '{origin_zone}', expected '{self.correct_starting_zone}'.")

            elif len(missing_ids) > 1:
                self._log_error(self.ERROR_MULTI_ACTIVE, f"Multiple syringes ({missing_ids}) missing simultaneously. Ambiguous pickup.")

            else: # Check for unexpected insertions
                for det in detections_by_source['manikin']:
                    if det['zone'] in self.target_zones:
                         # Check if it JUST entered
                         last_zone = self.syringe_last_known_zone.get((det['id'], 'manikin'), "Unknown")
                         if last_zone not in self.target_zones:
                              self._log_error(self.ERROR_UNEXPECTED_INSERT, f"Syringe ID {det['id']} appeared unexpectedly in target zone '{det['zone']}' while IDLE.")

        elif self.current_state == self.STATE_SYRINGE_PICKED:
            pickup_id = self.active_syringe_id # ID from syringe cam

            # 1. Check for insertion (ANY syringe in target zone)
            insertion_detected = None
            for det in detections_by_source['manikin']:
                if det['zone'] in self.target_zones:
                    insertion_detected = det
                    break # Take the first one found in a target zone

            if insertion_detected:
                insertion_id = insertion_detected['id'] # Manikin cam ID
                insertion_zone = insertion_detected['zone']
                self._log(f"Syringe detected in target zone '{insertion_zone}' (Manikin Cam ID: {insertion_id}). Assuming insertion of pickup ID {pickup_id}. State -> {self.STATE_SYRINGE_INSERTED}.")

                self.active_syringe_id = insertion_id # Switch to manikin ID
                self.active_id_source = 'manikin'
                self.active_syringe_current_zone = insertion_zone
                self.current_state = self.STATE_SYRINGE_INSERTED
                self.last_state_update_time = current_time
                self.volume_determined_this_cycle = False # Reset volume flag on new insertion
                self.active_syringe_volume = None # Reset stored volume

                # Perform TARGET ZONE check immediately
                if insertion_zone != self.correct_target_zone:
                    self._log_error(self.ERROR_WRONG_TARGET, f"Inserted into WRONG zone '{insertion_zone}', expected '{self.correct_target_zone}'.")
                else:
                    self._log(f"Inserted into CORRECT target zone '{insertion_zone}'.")

                if self.ERROR_WRONG_SYRINGE in self.error_flags_this_cycle:
                    self._log("Reminder: Syringe was picked from wrong start zone.", level="WARN")

                # Attempt initial volume check (but don't log error if undetermined yet)
                volume = insertion_detected['volumes'].get(self.correct_diameter, float('nan'))
                if volume is not None and not math.isnan(volume) and volume >= 0:
                    self.active_syringe_volume = volume # Store the measured volume
                    self._log(f"Volume measured at insertion: {volume:.2f} mL (for {self.correct_diameter}cm diameter).") # Log measured volume
                    # Check if volume is within tolerance
                    if not (self.target_volume - self.volume_tolerance <= volume <= self.target_volume + self.volume_tolerance):
                        self._log_error(self.ERROR_WRONG_VOLUME, f"Initial Volume {volume:.2f}mL outside target {self.target_volume:.2f} +/- {self.volume_tolerance:.2f}mL.")
                    else:
                        self._log("Initial Volume is within target range.")
                    self.volume_determined_this_cycle = True # Mark as found
                else:
                     self._log(f"Volume not determined on initial insertion frame. Will keep checking.", level="DEBUG")

                return # End processing for this frame

            # 2. Check for premature return (Original pickup ID reappears on table)
            if pickup_id in current_syringe_cam_detections and current_syringe_cam_detections[pickup_id]['zone'] in self.table_zones:
                 returned_zone = current_syringe_cam_detections[pickup_id]['zone']
                 self._log_error(self.ERROR_PREMATURE_RETURN, f"Pickup ID {pickup_id} returned to table zone '{returned_zone}' before insertion.")
                 self.syringes_on_table[pickup_id] = returned_zone # Add back
                 self._reset_state(f"Syringe returned prematurely to '{returned_zone}'")
                 return

            # 3. Check for timeout - REMOVED

        elif self.current_state == self.STATE_SYRINGE_INSERTED:
            inserted_id = self.active_syringe_id # ID from manikin cam

            # 0. PERSISTENT VOLUME CHECK (if not already found)
            if not self.volume_determined_this_cycle:
                current_insertion_det = None
                for det in detections_by_source['manikin']:
                    if det['id'] == inserted_id:
                        current_insertion_det = det
                        break

                if current_insertion_det:
                    volume = current_insertion_det['volumes'].get(self.correct_diameter, float('nan'))
                    if volume is not None and not math.isnan(volume) and volume >= 0:
                        self.active_syringe_volume = volume
                        self._log(f"Volume determined (persistent check): {volume:.2f} mL (for {self.correct_diameter}cm diameter).")
                        # Check tolerance (only log error once)
                        if not (self.target_volume - self.volume_tolerance <= volume <= self.target_volume + self.volume_tolerance):
                            # Add error flag only if not already added
                            if self.ERROR_WRONG_VOLUME not in self.error_flags_this_cycle:
                                self._log_error(self.ERROR_WRONG_VOLUME, f"Determined volume {volume:.2f}mL outside target {self.target_volume:.2f} +/- {self.volume_tolerance:.2f}mL.")
                        else:
                             self._log("Determined volume is within target range.")
                        self.volume_determined_this_cycle = True # Mark as determined
                    # else: Still not determined, do nothing, try again next frame
                else:
                    self._log(f"Inserted syringe {inserted_id} not visible for persistent volume check this frame.", level="DEBUG")


            # 1. Check for return (ANY syringe appears in table zone that wasn't there before)
            return_detected = None
            for det in detections_by_source['syringes']:
                if det['zone'] in self.table_zones and det['id'] not in self.syringes_on_table:
                    return_detected = det
                    break # Assume first one is the return

            if return_detected:
                returned_id = return_detected['id'] # Syringe cam ID
                returned_zone = return_detected['zone']
                self._log(f"Syringe ID {returned_id} appeared in table zone '{returned_zone}'. Assuming return of inserted syringe (Manikin ID {inserted_id}).")
                self.syringes_on_table[returned_id] = returned_zone # Update table state

                 # Check if volume was ever determined before logging final status
                if not self.volume_determined_this_cycle:
                     self._log("Volume was never determined before syringe return.", level="WARN")
                     # Optionally add a specific error flag here if needed
                     # self.error_flags_this_cycle.add("Volume Never Determined")

                # Check if cycle was successful before setting completion flag
                if not self.error_flags_this_cycle:
                    self._log("Workflow cycle completed successfully.", level="INFO")
                    self.workflow_completed = True # Signal completion
                    self._reset_state(f"Cycle completed successfully, syringe returned to '{returned_zone}'")
                else:
                    self._reset_state(f"Cycle completed with errors, syringe returned to '{returned_zone}'")
                return # Exit update_state

            # 2. Check if inserted syringe still visible on manikin cam (for logging movement)
            inserted_syringe_still_visible = False
            for det in detections_by_source['manikin']:
                 if det['id'] == inserted_id:
                     inserted_syringe_still_visible = True
                     if det['zone'] != self.active_syringe_current_zone: # Log if it moves
                          if DEBUG_MODE: self._log(f"Inserted syringe {inserted_id} moved to '{det['zone']}'", level="DEBUG")
                          self.active_syringe_current_zone = det['zone']
                     break

            # 3. Check for timeout - REMOVED


    def get_log_summary(self) -> str:
        return "\n".join(self._log_entries)

# --- End of SyringeTestWorkflow Class ---


# --- Main Execution Script ---
if __name__ == "__main__":

    print("--- Syringe Workflow Test Initializing (Dual Webcam Mode - Simplified Logic v2 - No Timeout / Persistent Vol) ---")
    print(f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"DEBUG MODE: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")


    # ----- Configuration: USER MUST EDIT THESE VALUES -----
    YOLO_MODEL_PATH = "runs/pose/train-pose11n-v32/weights/best.pt" # *** EDIT HERE ***
    POSSIBLE_SYRINGE_DIAMETERS_CM = [1.0, 1.25, 2.0] # *** EDIT HERE ***
    MANIKIN_CAMERA_INDEX = 1    # *** EDIT HERE ***
    SYRINGES_CAMERA_INDEX = 0   # *** EDIT HERE ***

    MANIKIN_TARGET_ZONE_NAMES = ["Arm", "Throat", "Foot"] # *** EDIT HERE ***
    MANIKIN_ZONE_DEFINITIONS = [
        ActiveZone(name="Arm", rect=(1250, 500, 1500, 850)),
        ActiveZone(name="Throat", rect=(1550, 300, 1750, 600)),
        ActiveZone(name="Foot", rect=(550, 450, 800, 700)) 
    ]

    SYRINGE_TABLE_ZONE_NAMES = ["Table Zone 1", "Table Zone 2", "Table Zone 3"] # *** EDIT HERE ***
    SYRINGE_FRAME_W, SYRINGE_FRAME_H = 1920, 1080 # Example frame size for coordinate reference
    gap=10; zone_width=420; zone_height=600; bottom=SYRINGE_FRAME_H-50; top=bottom-zone_height
    SYRINGE_ZONE_DEFINITIONS = [
        ActiveZone(name="Table Zone 1", rect=(800, 500, 1350, 1200)), # *** EDIT HERE ***
        ActiveZone(name="Table Zone 2", rect=(1350, 500, 1700, 1200)), # *** EDIT HERE ***
        ActiveZone(name="Table Zone 3", rect=(1700, 500, 2400, 1200)), # *** EDIT HERE ***
        #ActiveZone(name="Table Zone 4", rect=(100 + 3 * (zone_width + gap), top, 100 + 4 * zone_width + 3 * gap, bottom)), # *** EDIT HERE ***
    ]

    CORRECT_STARTING_ZONE = "Table Zone 3"      # *** EDIT HERE ***
    CORRECT_SYRINGE_DIAMETER_CM = 1.00           # *** EDIT HERE ***
    TARGET_VOLUME_ML = 2                     # *** EDIT HERE ***
    VOLUME_TOLERANCE_ML = 1                  # *** EDIT HERE ***
    CORRECT_TARGET_ZONE = "Arm"             # *** EDIT HERE ***

    # --- Timeouts Removed ---
    # PICKUP_INSERT_TIMEOUT = 15.0
    # INSERT_RETURN_TIMEOUT = 25.0
    # SYRINGE_PURGE_TIMEOUT = 20.0

    SAVE_OUTPUT_VIDEO = True # *** EDIT HERE ***
    OUTPUT_VIDEO_PATH_MANIKIN = "manikin_processed.mp4"
    OUTPUT_VIDEO_PATH_SYRINGES = "syringes_processed.mp4"
    RAW_CSV_PATH = 'syringe_volume_data_raw_combined.csv'
    WORKFLOW_LOG_PATH = 'syringe_test_workflow_log.txt'
    DEVICE_PREF_MANIKIN = None # *** EDIT HERE *** e.g., "cuda", "mps", "cpu" or None for auto
    DEVICE_PREF_SYRINGES = None # *** EDIT HERE ***
    # ----- End of User Configuration -----

    # --- Config Validation ---
    print("Validating configuration...")
    all_zone_defs = MANIKIN_ZONE_DEFINITIONS + SYRINGE_ZONE_DEFINITIONS
    all_zone_names = [z.name for z in all_zone_defs]
    if len(all_zone_names) != len(set(all_zone_names)): print(f"FATAL: Duplicate zone names: {all_zone_names}"); sys.exit(1)
    required_table = set(SYRINGE_TABLE_ZONE_NAMES); defined_syringe = set(z.name for z in SYRINGE_ZONE_DEFINITIONS)
    if not required_table.issubset(defined_syringe): print(f"FATAL: Missing syringe table zones: {required_table - defined_syringe}"); sys.exit(1)
    required_target = set(MANIKIN_TARGET_ZONE_NAMES); defined_manikin = set(z.name for z in MANIKIN_ZONE_DEFINITIONS)
    if not required_target.issubset(defined_manikin): print(f"FATAL: Missing manikin target zones: {required_target - defined_manikin}"); sys.exit(1)
    if CORRECT_STARTING_ZONE not in required_table: print(f"FATAL: Correct start zone '{CORRECT_STARTING_ZONE}' not in table zones."); sys.exit(1)
    if CORRECT_TARGET_ZONE not in required_target: print(f"FATAL: Correct target zone '{CORRECT_TARGET_ZONE}' not in target zones."); sys.exit(1)
    if CORRECT_SYRINGE_DIAMETER_CM not in POSSIBLE_SYRINGE_DIAMETERS_CM: print(f"FATAL: Correct diameter {CORRECT_SYRINGE_DIAMETER_CM} not possible."); sys.exit(1)
    if MANIKIN_CAMERA_INDEX == SYRINGES_CAMERA_INDEX: print(f"FATAL: Cameras use same index ({MANIKIN_CAMERA_INDEX})."); sys.exit(1)
    print("Configuration valid.")

    # --- Initialize ---
    estimator_manikin=None; estimator_syringes=None; workflow=None
    cap_manikin=None; cap_syringes=None; out_manikin=None; out_syringes=None
    csvfile=None; writer=None
    try:
        estimator_manikin=SyringeVolumeEstimator(YOLO_MODEL_PATH,POSSIBLE_SYRINGE_DIAMETERS_CM,MANIKIN_ZONE_DEFINITIONS,device_preference=DEVICE_PREF_MANIKIN)
        estimator_syringes=SyringeVolumeEstimator(YOLO_MODEL_PATH,POSSIBLE_SYRINGE_DIAMETERS_CM,SYRINGE_ZONE_DEFINITIONS,device_preference=DEVICE_PREF_SYRINGES)
        workflow = SyringeTestWorkflow(
            table_zone_names=SYRINGE_TABLE_ZONE_NAMES,
            target_zone_names=MANIKIN_TARGET_ZONE_NAMES,
            correct_starting_zone=CORRECT_STARTING_ZONE,
            correct_syringe_diameter=CORRECT_SYRINGE_DIAMETER_CM,
            possible_diameters=POSSIBLE_SYRINGE_DIAMETERS_CM,
            target_volume_ml=TARGET_VOLUME_ML,
            volume_tolerance_ml=VOLUME_TOLERANCE_ML,
            correct_target_zone=CORRECT_TARGET_ZONE,
            log_file_path=WORKFLOW_LOG_PATH,
            # Timeouts removed from instantiation
        )

        print("Setting up captures..."); cap_manikin=cv2.VideoCapture(MANIKIN_CAMERA_INDEX); cap_syringes=cv2.VideoCapture(SYRINGES_CAMERA_INDEX)
        if not cap_manikin.isOpened(): raise IOError(f"Cannot open manikin cam: {MANIKIN_CAMERA_INDEX}")
        if not cap_syringes.isOpened(): raise IOError(f"Cannot open syringes cam: {SYRINGES_CAMERA_INDEX}")
        # Attempt to set high resolution, but use what we get
        cap_manikin.set(cv2.CAP_PROP_FRAME_WIDTH, 9999); cap_manikin.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
        manikin_w=int(cap_manikin.get(cv2.CAP_PROP_FRAME_WIDTH)); manikin_h=int(cap_manikin.get(cv2.CAP_PROP_FRAME_HEIGHT)); manikin_fps=cap_manikin.get(cv2.CAP_PROP_FPS) if cap_manikin.get(cv2.CAP_PROP_FPS)>0 else 30.0
        print(f"Manikin cam {MANIKIN_CAMERA_INDEX}: {manikin_w}x{manikin_h} @ {manikin_fps:.1f} FPS")
        cap_syringes.set(cv2.CAP_PROP_FRAME_WIDTH, 9999); cap_syringes.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
        syringes_w=int(cap_syringes.get(cv2.CAP_PROP_FRAME_WIDTH)); syringes_h=int(cap_syringes.get(cv2.CAP_PROP_FRAME_HEIGHT)); syringes_fps=cap_syringes.get(cv2.CAP_PROP_FPS) if cap_syringes.get(cv2.CAP_PROP_FPS)>0 else 30.0
        print(f"Syringes cam {SYRINGES_CAMERA_INDEX}: {syringes_w}x{syringes_h} @ {syringes_fps:.1f} FPS")

        if SAVE_OUTPUT_VIDEO:
            estimator_manikin.save_video=True; estimator_syringes.save_video=True; print("Setting up writers...")
            out_manikin=estimator_manikin.setup_video_writer(OUTPUT_VIDEO_PATH_MANIKIN,manikin_w,manikin_h,manikin_fps)
            out_syringes=estimator_syringes.setup_video_writer(OUTPUT_VIDEO_PATH_SYRINGES,syringes_w,syringes_h,syringes_fps)
            if out_manikin is None: estimator_manikin.save_video=False
            if out_syringes is None: estimator_syringes.save_video=False

        print(f"Logging raw data to: {RAW_CSV_PATH}"); os.makedirs(os.path.dirname(RAW_CSV_PATH) or '.', exist_ok=True)
        file_exists=os.path.exists(RAW_CSV_PATH) and os.path.getsize(RAW_CSV_PATH) > 0; csvfile=open(RAW_CSV_PATH,'a',newline=''); writer=csv.writer(csvfile)
        if not file_exists: header=['timestamp','camera_source','track_id','center_x','center_y']+[f'volume_D{D:.2f}cm' for D in POSSIBLE_SYRINGE_DIAMETERS_CM]+['zone_name','in_active_zone_flag']; writer.writerow(header)

        print("\n--- Starting Main Loop (Press 'q' to quit) ---"); frame_count=0; start_time=time.monotonic()
        while True:
            ret_m, frame_m = cap_manikin.read(); ret_s, frame_s = cap_syringes.read()
            if not ret_m: print("Manikin camera read error. Exiting."); break
            if not ret_s: print("Syringes camera read error. Exiting."); break

            current_loop_time = time.monotonic(); timestamp = current_loop_time; frame_count += 1

            # Process frames even if previous had errors
            try:
                annotated_frame_m, detections_m = estimator_manikin.process_frame(frame_m, timestamp, writer, 'manikin')
            except Exception as e:
                 print(f"Error processing manikin frame: {e}")
                 annotated_frame_m = frame_m # Show raw frame on error
                 detections_m = []
                 # Optionally add a visual indicator to the frame
                 cv2.putText(annotated_frame_m, "MANIKIN FRAME ERROR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            try:
                annotated_frame_s, detections_s = estimator_syringes.process_frame(frame_s, timestamp, writer, 'syringes')
            except Exception as e:
                 print(f"Error processing syringes frame: {e}")
                 annotated_frame_s = frame_s # Show raw frame on error
                 detections_s = []
                 cv2.putText(annotated_frame_s, "SYRINGES FRAME ERROR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            all_detections = detections_m + detections_s
            workflow.update_state(all_detections, current_loop_time)

            # --- Check for workflow completion to auto-exit ---
            if workflow.workflow_completed:
                print("Workflow marked as complete. Exiting loop.")
                break # Exit the main loop

            if out_manikin and estimator_manikin.save_video: out_manikin.write(annotated_frame_m)
            if out_syringes and estimator_syringes.save_video: out_syringes.write(annotated_frame_s)

            # Display
            try:
                h_m,w_m=annotated_frame_m.shape[:2]; h_s,w_s=annotated_frame_s.shape[:2]; max_h=max(h_m,h_s)
                # Resize frames to have the same height for horizontal stacking
                if h_m == 0 or h_s == 0: continue # Skip display if a frame is invalid
                display_m=cv2.resize(annotated_frame_m,(int(w_m*max_h/h_m),max_h)) if h_m!=max_h else annotated_frame_m
                display_s=cv2.resize(annotated_frame_s,(int(w_s*max_h/h_s),max_h)) if h_s!=max_h else annotated_frame_s

                combined=np.hstack((display_m,display_s))

                # Resize combined image if too wide for screen
                max_w_display=1920 # Max width for display window
                if combined.shape[1]>max_w_display:
                    scale=max_w_display/combined.shape[1]
                    combined=cv2.resize(combined,(max_w_display,int(combined.shape[0]*scale)))

                cv2.imshow('Syringe Workflow - Press Q to Quit', combined)
            except Exception as e:
                print(f"Display error: {e}")
                # Attempt to show individual frames if stacking fails
                try: cv2.imshow('Manikin Cam', cv2.resize(annotated_frame_m, (annotated_frame_m.shape[1]//2, annotated_frame_m.shape[0]//2))) # Smaller resize
                except: pass # Ignore error if manikin frame fails
                try: cv2.imshow('Syringes Cam', cv2.resize(annotated_frame_s, (annotated_frame_s.shape[1]//2, annotated_frame_s.shape[0]//2))) # Smaller resize
                except: pass # Ignore error if syringes frame fails


            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed. Exiting loop.")
                break

        end_time=time.monotonic(); print(f"--- Loop Ended: Processed {frame_count} frames in {end_time - start_time:.2f}s ---")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting.")
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR DETAILS: {e}")
        print("-" * 20)
        traceback.print_exc()
        print("-" * 20)
    finally:
        print("Releasing resources...");
        if cap_manikin: cap_manikin.release(); print("Manikin capture released.")
        if cap_syringes: cap_syringes.release(); print("Syringes capture released.")
        if out_manikin: out_manikin.release(); print("Manikin writer released.")
        if out_syringes: out_syringes.release(); print("Syringes writer released.")
        if csvfile and not csvfile.closed: csvfile.close(); print("CSV file closed.")
        cv2.destroyAllWindows(); print("Windows closed.")
        if workflow: print(f"\n--- Final Workflow Log ({WORKFLOW_LOG_PATH}) ---"); print(workflow.get_log_summary()); print("-" * 50)
        print("--- Script Finished ---")