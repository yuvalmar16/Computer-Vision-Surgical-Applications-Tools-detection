# Surgical Tool Detection using YOLO with Semi-Supervised Learning

This project involves the use of YOLO (You Only Look Once) to train a model on an image dataset, predict pseudo labels on In-Distribution (ID) videos, fine-tune the model, and then predict on Out-Of-Distribution (OOD) videos. The project leverages semi-supervised learning techniques to improve the detection accuracy of surgical tools.

## Project Structure

- **predict.py**: Functions to run predictions on images.
- **video.py**: Functions to run predictions on videos.
- **train.py**: Functions to train the model on labeled data and fine-tune using pseudo-labeled data.
- **yaml_files**
  - `train_labeled.yaml`: Configuration file for the In-Distribution dataset.
  - `train_labeled.yaml`: Configuration file for the labeled image dataset.
- **requirements.txt**: Lists the required Python packages.


## Installation

1. Clone the repository:
   ```sh
   git clone  https://github.com/yuvalmar16/Computer-Vision-Surgical-Applications-Tools-detection.git
   cd ComputerVisionOR
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

A link to the final model weights I used - [model_finetuned.pt](https://drive.google.com/file/d/1ikFNuP6OKctOnAODgeUitsjJUWMow2XJ/view?usp=sharing)

### 2. Predicting on Images

To run predictions on images:

```sh
python predict.py -i <image_path> -c <confidence_level>
```

or

```sh
python predict.py -d <images_folder> -c <confidence_level>
```

Note that the -c parameter is optional, and if not specified, the default value is 0.8.

### 3. Predicting on Videos

To run predictions on videos:

```sh
python video.py -v <video_path> -c <confidence_level>
```

or

```sh
python video.py -d <videos_folder> -c <confidence_level>
```

Note that the -c parameter is optional, and if not specified, the default value is 0.7.

## File Descriptions

- **yaml_files/id.yaml**: Configuration for the In-Distribution dataset.
- **yaml_files/train_labeled.yaml**: Configuration for the labeled image data.
- **main.py**: Main script to control the workflow.
- **predict.py**: Script for running predictions on images.
- **train.py**: Contains training and fine-tuning functions.
- **video.py**: Script for running predictions on videos.
