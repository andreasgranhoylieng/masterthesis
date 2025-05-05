import cv2
import time
import datetime
import os

# --- Configuration ---
camera_index_1 = 0  # First camera index
camera_index_2 = 1  # Second camera index

# Desired Frame Rate (FPS) - Adjust if needed
desired_fps = 20.0

# Video output directory
output_dir = "recordings"
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Video codec (FourCC) for MP4:
# 'mp4v' - MPEG-4 codec, generally compatible (recommended to try first)
# 'avc1' or 'h264' - H.264/AVC codec (often better compression, but may need specific backend support like FFmpeg)
# 'MJPG' - Motion JPEG (can be put in MP4, but less common, larger files)
# Choose the codec:
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# If 'mp4v' causes issues, try:
# fourcc = cv2.VideoWriter_fourcc(*'avc1') # Requires H.264 support
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
    if cap1.isOpened(): cap1.release() # Release the first camera if the second fails
    exit()

print("Cameras opened successfully.")

# Get frame width and height (use dimensions from the first camera)
# See previous notes about potential issues if camera resolutions differ.
ret1, frame1_test = cap1.read()
if not ret1:
    print("Error: Could not read frame from camera 1 to get dimensions.")
    cap1.release()
    cap2.release()
    exit()

frame_width = int(frame1_test.shape[1])
frame_height = int(frame1_test.shape[0])
print(f"Using frame dimensions (Width x Height): {frame_width} x {frame_height}")
print(f"Target FPS: {desired_fps}")
print(f"Using FourCC codec: {''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])}") # Decode FourCC for display

# --- Setup Video Writers ---
# Generate unique filenames using timestamps with .mp4 extension
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename1 = os.path.join(output_dir, f"cam{camera_index_1}_{timestamp}.mp4") # Changed extension
filename2 = os.path.join(output_dir, f"cam{camera_index_2}_{timestamp}.mp4") # Changed extension

print(f"Output file 1: {filename1}")
print(f"Output file 2: {filename2}")

# Initialize VideoWriter objects
out1 = cv2.VideoWriter(filename1, fourcc, desired_fps, (frame_width, frame_height))
out2 = cv2.VideoWriter(filename2, fourcc, desired_fps, (frame_width, frame_height))

# Check if VideoWriters initialized successfully
# This is often where codec issues will appear first
if not out1.isOpened():
    print(f"Error: Could not open VideoWriter for {filename1}.")
    print("Check if the selected FourCC codec ('mp4v' or other) is supported by your OpenCV installation/backend.")
    cap1.release()
    cap2.release()
    exit()
if not out2.isOpened():
    print(f"Error: Could not open VideoWriter for {filename2}.")
    print("Check if the selected FourCC codec ('mp4v' or other) is supported by your OpenCV installation/backend.")
    cap1.release()
    cap2.release()
    if out1.isOpened(): out1.release() # Release the first writer if the second fails
    exit()

print("\n--- Recording Started ---")
print("Press 'q' in the display window to stop recording.")

# --- Recording Loop ---
recording_start_time = time.time()
frame_count = 0

while True:
    # Read frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # If frames were read successfully, write and display
    if ret1 and ret2:
        # Write frames to the output files
        out1.write(frame1)
        out2.write(frame2)

        # Display the recordings (optional)
        display_frame1 = cv2.resize(frame1, (640, 480))
        display_frame2 = cv2.resize(frame2, (640, 480))
        cv2.imshow(f'Camera {camera_index_1}', display_frame1)
        cv2.imshow(f'Camera {camera_index_2}', display_frame2)

        frame_count += 1
    elif not ret1:
        print("Warning: Failed to grab frame from camera 1")
        # Decide how to handle missing frames (e.g., break, continue, write black frame?)
    elif not ret2:
        print("Warning: Failed to grab frame from camera 2")
        # Decide how to handle missing frames

    # Check for 'q' key press to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n'q' pressed, stopping recording...")
        break
    # Optional: Check if either camera capture failed decisively during loop
    # if not cap1.isOpened() or not cap2.isOpened():
    #    print("Error: Camera disconnected during recording.")
    #    break


# --- Cleanup ---
print("\n--- Releasing Resources ---")
recording_end_time = time.time()
duration = recording_end_time - recording_start_time
actual_fps = frame_count / duration if duration > 0 else 0

print(f"Recording duration: {duration:.2f} seconds")
print(f"Frames recorded: {frame_count}")
print(f"Actual average FPS: {actual_fps:.2f}")


# Release video capture and writer objects
# Crucial for finalizing the video files correctly!
cap1.release()
cap2.release()
out1.release()
out2.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Recording finished and files saved.")
print(f"Video 1 saved to: {filename1}")
print(f"Video 2 saved to: {filename2}")