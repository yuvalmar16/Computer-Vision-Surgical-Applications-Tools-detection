import os
import cv2
from ultralytics import YOLO

def run_yolo_on_video(
        model_path, 
        video_path, 
        output_dir, 
        conf_thresh=0.5, 
        fps=25, 
        make_video=True):

    #Load the model
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "frames")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    frame_size = None
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        frame_name = f"{frame_num:06}.jpg"
        img_save_path = os.path.join(img_dir, frame_name)
        label_save_path = os.path.join(label_dir, frame_name.replace('.jpg', '.txt'))

        # 3. Inference
        results = model.predict(frame, conf=conf_thresh, imgsz=640)[0]

        # 4. Visualization box
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.item())
                cls = int(box.cls.item())

               
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label_text = f"{cls} {conf:.2f}"
                cv2.putText(frame, label_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(img_save_path, frame)
        all_frames.append(img_save_path)

        # 6.YOLO Format: x_center, y_center, w, h, conf, class
        h, w = frame.shape[:2]
        with open(label_save_path, 'w') as f:
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf.item())
                    cls = int(box.cls.item())
                    # Calc YOLO format
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f} {conf:.6f} {cls}\n")

        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])

    cap.release()

    # 7.Build video from frames
    if make_video:
        output_video = os.path.join(output_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
        for img_path in all_frames:
            frame = cv2.imread(img_path)
            if frame.shape[:2] != (frame_size[1], frame_size[0]):
                frame = cv2.resize(frame, frame_size)
            video_writer.write(frame)
        video_writer.release()
        print(f"Video saved to {output_video}")

    print(f"Done! All frames and labels saved to {output_dir}")


if __name__ == "__main__":
    model_path = "/home/student/Desktop/HW1_surgical_vision/weights/best.pt"
    video_path = "/home/student/Desktop/HW1_surgical_vision/input_video.mp4"
    output_dir = "/home/student/Desktop/HW1_surgical_vision/OOD_predictions_visualized"
    conf_thresh = 0.65
    fps = 25 

    run_yolo_on_video(model_path, video_path, output_dir, conf_thresh, fps)
