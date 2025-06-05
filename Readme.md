# Surgical Tool Detection using YOLO with Semi-Supervised Learning

This project involves the use of YOLO (You Only Look Once) to train a model on an image dataset, predict pseudo labels on In-Distribution (ID) videos, fine-tune the model, and then predict on Out-Of-Distribution (OOD) videos. The project leverages semi-supervised learning techniques to improve the detection accuracy of surgical tools.

## Project Structure

- **predict.py**: Functions to run predictions on images.
- **video.py**: Functions to run predictions on videos.
- **train.py**: Functions to train the model on labeled data and fine-tune using pseudo-labeled data.
- **yaml_files**
  - yaml_files/surgical_pesudo.yaml: Configuration for the pesudo-labeled In-Distribution dataset with the labeled image data.
  - yaml_files/only_labeled.yaml: Configuration for the labeled image data.
- **requirements.txt**: Lists the required Python packages.


## Installation

1. Clone the repository:
   ```sh
   git clone  https://github.com/yuvalmar16/Computer-Vision-Surgical-Applications-Tools-detection.git
   cd Computer-Vision-Surgical-Applications-Tools-detection
   ```

2. Recommended -    
   Create a virtual environment:
   ```sh
   python -m venv cv_tool_detection
   source cv_hw1/bin/activate
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model

Train the model on the labeled image dataset:

```sh
python train.py
```

This will:
- Train the model for 100 epochs on the labeled image dataset.
- Predict pseudo labels on the ID videos.
- Fine-tune the model for 300 epochs using the pseudo-labeled data.
- Predict on the OOD videos with visualization.
(Note that you can change the number of epochs to your liking, it is set in the main.py file.)

A link to the final model weights I used - [best.pt] (https://github.com/yuvalmar16/Computer-Vision-Surgical-Applications-Tools-detection/blob/main/best.pt)

### 2. Predicting on Images

To run predictions on images:
update the path for the Image, the model weights("best.pt") and ouptput dir, and run
```sh
python predict.py 
```

### 3. Predicting on Videos

To run predictions on videos:
update the path for the video, the model weights("best.pt") and ouptput dir, and run
```sh
python video.py 
```

## File Descriptions

- **yaml_files/surgical_pesudo.yaml**: Configuration for the pesudo-labeled In-Distribution dataset with the labeled image data.
- **yaml_files/only_labeled.yaml**: Configuration for the labeled image data.
- **predict.py**: Script for running predictions on images.
- **train.py**: Contains training and fine-tuning functions.
- **video.py**: Script for running predictions on videos.
