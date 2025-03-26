import argparse
import os

import cv2


def extract_frames(video_path: str, output_dir: str, desired_fps: float) -> None:
    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_fps: float = cap.get(cv2.CAP_PROP_FPS)
    frame_interval: float = 1 / desired_fps  # seconds between frames

    next_capture_time: float = 0.0  # next timestamp to capture
    frame_count: int = 0
    saved_count: int = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time: float = frame_count / video_fps

        if current_time >= next_capture_time:
            # Save the frame with the same resolution.
            frame_filename = os.path.join(
                output_dir, f"frame_{saved_count:05d}.png"
            )
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            next_capture_time += frame_interval

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from video at specified fps."
    )
    parser.add_argument("video", type=str, help="Path to the video file")
    parser.add_argument(
        "output_dir", type=str, help="Directory to save extracted frames"
    )
    parser.add_argument(
        "fps", type=float, help="Number of frames per second to extract"
    )

    args = parser.parse_args()
    extract_frames(args.video, args.output_dir, args.fps)
