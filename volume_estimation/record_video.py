import cv2
import time
import datetime
import os

# --- Configuration ---
camera_index_1 = 0  # First camera index
camera_index_2 = 1  # Second camera index

# Desired Capture Resolution
desired_capture_width = 3840
desired_capture_height = 2160
# desired_capture_width = 1920 # Fallback to Full HD if 4K causes issues
# desired_capture_height = 1080

# Desired Frame Rate (FPS) - Adjust if needed
desired_fps = 20.0 # Note: Higher resolutions might force lower FPS on some cameras

# Video output directory
output_dir = "recordings"
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Video codec (FourCC) for MP4:
# 'mp4v' - MPEG-4 codec, generally compatible
# 'avc1' or 'h264' - H.264/AVC codec (often better compression, may need FFmpeg backend)
# 'XVID' - Another common MPEG-4 codec
# 'MJPG' - Motion JPEG
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# --- End Configuration ---

# --- Initialization ---
print("Initializing cameras...")
cap1 = cv2.VideoCapture(camera_index_1)
cap2 = cv2.VideoCapture(camera_index_2)

# Check if cameras opened successfully
if not cap1.isOpened():
    print(f"Error: Could not open camera {camera_index_1}")
    exit()
if not cap2.isOpened():
    print(f"Error: Could not open camera {camera_index_2}")
    if cap1.isOpened(): cap1.release()
    exit()

print("Cameras opened successfully.")

# --- Attempt to Set Camera Resolution ---
print(f"Attempting to set resolution to {desired_capture_width}x{desired_capture_height} for camera {camera_index_1}...")
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, desired_capture_width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_capture_height)
# Some cameras need a moment to apply settings, or a frame read
time.sleep(0.5) # Optional delay

print(f"Attempting to set resolution to {desired_capture_width}x{desired_capture_height} for camera {camera_index_2}...")
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, desired_capture_width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_capture_height)
time.sleep(0.5) # Optional delay

# --- Get Actual Frame Dimensions (from the first camera) ---
# This will tell us what resolution cap1 is actually providing after the set attempt.
ret1_test, frame1_test = cap1.read()
if not ret1_test:
    print("Error: Could not read frame from camera 1 to get dimensions after setting resolution.")
    cap1.release()
    cap2.release()
    exit()

frame_width = int(frame1_test.shape[1])
frame_height = int(frame1_test.shape[0])

# Verify actual resolution obtained for camera 1
actual_width_cam1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height_cam1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual resolution obtained for camera {camera_index_1}: {actual_width_cam1}x{actual_height_cam1}")

if frame_width != actual_width_cam1 or frame_height != actual_height_cam1 :
    print(f"Warning: Frame shape dimensions ({frame_width}x{frame_height}) "
          f"differ from CAP_PROP dimensions ({actual_width_cam1}x{actual_height_cam1}) for camera 1. "
          f"Using frame shape dimensions for VideoWriter.")
    # This can sometimes happen. The frame.shape is usually more reliable for VideoWriter.

# Check camera 2's actual resolution for informational purposes
actual_width_cam2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height_cam2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual resolution obtained for camera {camera_index_2}: {actual_width_cam2}x{actual_height_cam2}")

if actual_width_cam2 != frame_width or actual_height_cam2 != frame_height:
    print(f"Warning: Camera {camera_index_2} ({actual_width_cam2}x{actual_height_cam2}) "
          f"has a different resolution than camera {camera_index_1} ({frame_width}x{frame_height}) "
          f"which is used for both recordings. Frame from camera 2 might be resized or cause issues if not matching.")


print(f"Using frame dimensions for VideoWriters (Width x Height): {frame_width} x {frame_height}")
print(f"Target FPS: {desired_fps}")
print(f"Using FourCC codec: {''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])}")

# --- Setup Video Writers ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename1 = os.path.join(output_dir, f"cam{camera_index_1}_{timestamp}_{frame_width}x{frame_height}.mp4")
filename2 = os.path.join(output_dir, f"cam{camera_index_2}_{timestamp}_{frame_width}x{frame_height}.mp4")

print(f"Output file 1: {filename1}")
print(f"Output file 2: {filename2}")

out1 = cv2.VideoWriter(filename1, fourcc, desired_fps, (frame_width, frame_height))
out2 = cv2.VideoWriter(filename2, fourcc, desired_fps, (frame_width, frame_height))

if not out1.isOpened():
    print(f"Error: Could not open VideoWriter for {filename1}.")
    cap1.release()
    cap2.release()
    exit()
if not out2.isOpened():
    print(f"Error: Could not open VideoWriter for {filename2}.")
    cap1.release()
    cap2.release()
    if out1.isOpened(): out1.release()
    exit()

print("\n--- Recording Started ---")
print("Press 'q' in the display window to stop recording.")

# --- Recording Loop ---
recording_start_time = time.time()
frame_count = 0

try:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:
            # Ensure frame2 has the same dimensions as expected by out2
            # This is a basic check; if resolutions widely differ, resizing might be needed
            # or the VideoWriter might handle it (sometimes with stretching/cropping) or error out.
            if frame2.shape[0] != frame_height or frame2.shape[1] != frame_width:
                print(f"Warning: Frame from camera {camera_index_2} has dimensions {frame2.shape[1]}x{frame2.shape[0]}, "
                      f"but VideoWriter expects {frame_width}x{frame_height}. Attempting to resize.")
                try:
                    frame2_resized = cv2.resize(frame2, (frame_width, frame_height))
                    out2.write(frame2_resized)
                except Exception as e:
                    print(f"Error resizing or writing frame from camera 2: {e}")
                    # Decide how to handle: break, continue, write black frame etc.
                    # For now, we'll just skip writing this frame for cam2
                    pass # Or write frame2 directly if you want to see if it errors: out2.write(frame2)
            else:
                out2.write(frame2)

            out1.write(frame1) # frame1 is already the reference dimension

            # Display the recordings (optional, can be resource-intensive at high resolutions)
            # Consider displaying smaller resized frames to save CPU
            display_width = 640
            display_height = int(display_width * (frame_height / frame_width)) # maintain aspect ratio

            try:
                display_frame1 = cv2.resize(frame1, (display_width, display_height))
                cv2.imshow(f'Camera {camera_index_1}', display_frame1)
            except Exception as e:
                print(f"Error resizing/displaying frame 1: {e}")


            try:
                # If frame2 was resized for writing, use the original frame2 for display or the resized one
                # Using original frame2 for display to see its native look before potential resize for saving
                display_frame2_source = frame2
                if display_frame2_source.shape[1] != display_width or display_frame2_source.shape[0] != display_height :
                     display_frame2_temp_height = int(display_width * (display_frame2_source.shape[0] / display_frame2_source.shape[1]))
                     display_frame2 = cv2.resize(display_frame2_source, (display_width, display_frame2_temp_height))
                else:
                    display_frame2 = display_frame2_source

                cv2.imshow(f'Camera {camera_index_2}', display_frame2)
            except Exception as e:
                print(f"Error resizing/displaying frame 2: {e}")


            frame_count += 1
        elif not ret1:
            print("Warning: Failed to grab frame from camera 1. End of stream or error.")
            # break # Optional: stop if one camera fails
        elif not ret2:
            print("Warning: Failed to grab frame from camera 2. End of stream or error.")
            # break # Optional: stop if one camera fails

        if not ret1 and not ret2:
            print("Both cameras failed to provide frames. Stopping.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n'q' pressed, stopping recording...")
            break
        # Check if camera is still opened (might not catch all disconnects)
        if not cap1.isOpened() or not cap2.isOpened():
           print("Error: A camera became disconnected during recording.")
           break

except KeyboardInterrupt:
    print("\nRecording interrupted by user (Ctrl+C).")

# --- Cleanup ---
finally:
    print("\n--- Releasing Resources ---")
    recording_end_time = time.time()
    duration = recording_end_time - recording_start_time if recording_start_time else 0

    if duration > 0:
        actual_fps = frame_count / duration
        print(f"Recording duration: {duration:.2f} seconds")
        print(f"Frames recorded: {frame_count} (for cam1, assuming cam2 is similar)")
        print(f"Actual average FPS: {actual_fps:.2f}")
    else:
        print("No frames recorded or recording time was zero.")

    if cap1.isOpened(): cap1.release()
    if cap2.isOpened(): cap2.release()
    if out1.isOpened(): out1.release()
    if out2.isOpened(): out2.release()

    cv2.destroyAllWindows()

    print("Recording finished and files saved (if any frames were processed).")
    if frame_count > 0 :
        print(f"Video 1 saved to: {filename1}")
        print(f"Video 2 saved to: {filename2}")
    else:
        print(f"No video files were effectively saved as no frames were processed.")
        # Optionally remove empty files
        if os.path.exists(filename1) and os.path.getsize(filename1) < 1024 : # Check if file is very small
             try: os.remove(filename1)
             except OSError: pass
        if os.path.exists(filename2) and os.path.getsize(filename2) < 1024 :
             try: os.remove(filename2)
             except OSError: pass