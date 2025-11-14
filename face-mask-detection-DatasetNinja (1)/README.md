# Face Mask Detection and Masked Facial Recognition

This project implements a comprehensive solution for face mask detection and masked facial recognition using deep learning techniques. It combines object detection for identifying faces and classifying mask-wearing status with binary classification for mask compliance verification.

## Overview

The project utilizes YOLOv8 for real-time object detection to classify faces into three categories: "with_mask", "without_mask", and "mask_weared_incorrect". Additionally, it includes a binary classification pipeline that crops detected faces and uses various CNN architectures to determine mask compliance.

## Dataset

The dataset is sourced from DatasetNinja and consists of annotated images for face mask detection. The project processes and splits the data into train, validation, and test sets.

- **Classes**: with_mask, without_mask, mask_weared_incorrect
- **Splits**: Train (80%), Validation (10%), Test (10%)
- **Format**: YOLO annotation format (.txt files with normalized coordinates)

## Models and Training

### Object Detection (YOLOv8)
- **Models Trained**:
  - YOLOv8 Nano (yolov8n.pt): 15 epochs, batch size 32
  - YOLOv8 Small (yolov8s.pt): 60 epochs, batch size 16
- **Training Parameters**: lr0=0.002, weight_decay=5e-4, data augmentation (mosaic, mixup, etc.)
- **Inference**: Optimized for imgsz=960, conf=0.15, iou=0.6

### Binary Classification (Mask Compliance)
The project benchmarks 10 different CNN architectures for binary classification on cropped face images:

1. **DeepMaskNet** (Custom): Compact CNN with stem-body-head architecture
2. **GoogleNet**: Pretrained on ImageNet
3. **SqueezeNet**: Efficient architecture
4. **DenseNet201**: Deep dense connections
5. **ShuffleNet**: Lightweight mobile-friendly model
6. **ResNet18**: Classic residual network
7. **MobileNetV2**: Mobile-optimized CNN
8. **InceptionResNetV2**: Advanced inception-residual hybrid
9. **DarkNet19**: YOLO backbone variant
10. **DarkNet53**: Deeper DarkNet architecture

**Training Details**:
- **Epochs**: 8
- **Learning Rate**: 1e-3 (AdamW optimizer)
- **Image Size**: 224x224
- **Augmentation**: RandomResizedCrop, HorizontalFlip, ColorJitter (train only)
- **Balanced Sampling**: WeightedRandomSampler for class balance

## Evaluation Metrics

### Object Detection
- mAP (mean Average Precision)
- Precision, Recall, F1-Score per class
- Inference speed and model size analysis

### Binary Classification
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrices**: TP, FP, FN, TN for each model
- **Visualizations**: Bar charts, radar plots comparing all models

## Results Summary

The project provides comprehensive benchmarking results showing performance comparison across all tested models. DeepMaskNet and other architectures are evaluated for their effectiveness in mask compliance classification.

## Installation and Usage

### Prerequisites
```bash
pip install ultralytics>=8.2.0 opencv-python pillow timm torchmetrics
```

### Running the Notebook
1. Open `dlminiproj.ipynb` in Jupyter or Google Colab
2. Ensure dataset paths are correctly set (adjust for Kaggle/working or local environment)
3. Run cells sequentially to:
   - Prepare dataset from DatasetNinja
   - Train YOLO models
   - Perform inference and visualization
   - Train and benchmark classification models
   - Generate evaluation plots and confusion matrices

### Model Export
- YOLO models can be exported to ONNX format for deployment
- Classification models are saved as PyTorch checkpoints

## Project Structure
```
├── dlminiproj.ipynb                 # Main notebook
├── data/face_mask_dn/               # Processed dataset
│   ├── images/                      # Train/val/test images
│   └── labels/                      # YOLO annotations
├── runs_face_mask/                  # YOLO training runs
├── face-mask-detection-DatasetNinja (1)/  # Dataset source
└── A novel DeepMaskNet model...pdf  # Research paper
```

## Key Features
- Real-time face mask detection with YOLOv8
- Multi-model benchmarking for mask compliance
- Comprehensive evaluation with visualizations
- ONNX export for deployment
- Balanced dataset handling
- Custom DeepMaskNet architecture

## License
This project uses the DatasetNinja dataset. Please refer to the LICENSE.md file for dataset-specific licensing information.
