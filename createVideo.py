#!/usr/bin/env python3
import cv2
import os
import argparse

def make_video(input_dir: str, output_file: str, fps: float):
    # 1. Collect and sort all .png files
    images = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    if not images:
        raise ValueError(f"No PNG files found in {input_dir}")
    images.sort()  # zero-padded names sort correctly

    # 2. Read the first frame to get dimensions
    first_frame_path = os.path.join(input_dir, images[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        raise IOError(f"Could not read image {first_frame_path}")
    height, width, channels = frame.shape

    # 3. Set up VideoWriter
    #    'mp4v' gives .mp4 output; use 'XVID' for .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 4. Loop through and write each frame
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: skipping unreadable frame {img_name}")
            continue
        video_writer.write(frame)

    # 5. Clean up
    video_writer.release()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine a sequence of PNG frames into a video."
    )
    parser.add_argument(
        "input_dir",
        help="Path to the directory containing frame_####.png files"
    )
    parser.add_argument(
        "-o", "--output",
        default="output.mp4",
        help="Output video filename (default: output.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second for the output video (default: 24.0)"
    )

    args = parser.parse_args()
    make_video(args.input_dir, args.output, args.fps)
