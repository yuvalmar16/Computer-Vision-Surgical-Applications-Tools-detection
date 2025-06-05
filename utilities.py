import cv2
import os

def extract_frames_from_video(video_path, output_dir, frame_rate=1):
    """
    Extracts frames from a video at a specified frame rate and saves them in the output_dir.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved.
        frame_rate (int): Save every nth frame (default 1 = save all).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_rate == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1
    cap.release()
    print(f"Extracted {saved_idx} frames from {video_path} to {output_dir}")

# Example usage:
if __name__ == "__main__":
    # Just for quick testing—replace with your paths
    extract_frames_from_video("my_video.mp4", "frames_output", frame_rate=1)
