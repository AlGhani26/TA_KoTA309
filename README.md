# Semantic Segmentation of Satellite Images using UNet with SKResNeXt50 Backbone

## Project Description
This project implements semantic segmentation of satellite images for land cover classification using a modified UNet architecture with an SKResNeXt50 backbone. The model classifies satellite images into 6 land cover classes: Background, Built-up, Farmland, Forest, Meadow, and Water.

## Features
- Data preprocessing and augmentation for satellite imagery.
- Training with multiple model variants including baseline and modified architectures.
- Inference on test satellite images with visualization of predicted masks.
- Evaluation using metrics such as accuracy, precision, recall, F1 score, and mean Intersection over Union (mIoU).
- Visualization of predictions alongside ground truth masks.
- Measurement and logging of inference time performance.

## Installation
Install the required Python dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- matplotlib
- numpy
- Pillow
- rasterio
- scikit_learn
- seaborn
- kornia

## Dataset
- Satellite images are 16-bit TIFF files with 4 channels (NIR, R, G, B).
- Segmentation masks are 8-bit PNG images with 6 classes.
- Dataset is split into 80% training and 20% validation sets.
- Test data and labels are provided separately for inference evaluation.

## Folder Structure
- `ape_inference/`: Contains inference scripts, model architecture definitions, and notebooks.
- `ape_train/`: Contains training scripts, notebooks, datasets, and output models.
- `ape_preprocessing/`: Contains scripts for image preprocessing and patch extraction.
- `ape_inference/data_test/`: Test images and corresponding labels for inference.
- `ape_inference/data_visualisasi/`: Images and masks used for visualization.
- `ape_train/output/`: Output directory for trained models, logs, checkpoints, and inference results.
- `ape_train/dataset`: Dataset images and corresponding labels for training. 

## Usage

### Training
- Use the notebook `ape_train/train_baseline.ipynb` or `ape_train/train_modifikasi.ipynb` to train the models.
- Training includes data augmentation, model checkpointing, and logging.
- Models are trained for 120 epochs by default with Adam optimizer and CrossEntropyLoss.

### Inference
- Use the notebook `ape_inference/Inference.ipynb` to run inference on test images.
- Load pretrained models from the `ape_train/output/` directory.
- Visualize predictions alongside ground truth masks.
- Evaluate model performance with detailed metrics.

### Visualization
- Visualize satellite images, ground truth masks, and predicted masks using provided visualization functions.
- Visualizations include color-coded segmentation masks for easy interpretation.

## Evaluation Metrics
- Overall accuracy: Pixel-wise classification accuracy.
- Precision, Recall, and F1 score per class.
- Mean Intersection over Union (mIoU) for segmentation quality.
- Confusion matrix analysis.
- Inference time measurement for performance benchmarking.

## License
This project does not include a specific license. Please contact the author for usage permissions.

## Contact
For questions or collaboration, please reach out to the project maintainer.
