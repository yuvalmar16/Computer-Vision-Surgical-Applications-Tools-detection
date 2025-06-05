import os
import cv2
from ultralytics import YOLO

# === CONFIGURATION ===
MODEL_PATH = "/home/student/Desktop/HW1_surgical_vision/best.pt"
IMAGE_PATH = "/home/student/Desktop/HW1_surgical_vision/test_images/img01.jpg"
OUTPUT_DIR = "/home/student/Desktop/HW1_surgical_vision/prediction_output/"
CONFIDENCE_THRESHOLD = 0.5

def predict_on_image(model_path, image_path, output_dir, conf_thresh=0.5):
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(image_path)
    label_name = img_name.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(output_dir, label_name)
    save_img_path = os.path.join(output_dir, img_name)

    # Load model
    model = YOLO(model_path)
    # Read image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Predict
    results = model.predict(img, conf=conf_thresh, imgsz=640)[0]

    # Draw boxes and save labels
    with open(label_path, 'w') as f:
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.item())
                cls = int(box.cls.item())
                # YOLO format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f} {conf:.6f} {cls}\n")
                # Draw
                color = (0, 255, 0)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label_text = f"{cls} {conf:.2f}"
                cv2.putText(img, label_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                print(f"Class: {cls}, Confidence: {conf:.2f}, (x_center: {x_center:.3f}, y_center: {y_center:.3f}, w: {bw:.3f}, h: {bh:.3f})")

    cv2.imwrite(save_img_path, img)
    print(f" Saved image with predictions: {save_img_path}")
    print(f" Saved label file: {label_path}")

if __name__ == "__main__":
    predict_on_image(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR, CONFIDENCE_THRESHOLD)
