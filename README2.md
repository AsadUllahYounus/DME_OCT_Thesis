# SimCLR OCT Linear Evaluation

This repository contains code for linear evaluation of SimCLR models trained on OCT (Optical Coherence Tomography) images. The linear evaluation protocol is a standard method for assessing the quality of learned representations by training a linear classifier on frozen features.

## Overview

The codebase provides two main scripts for linear evaluation:
- `linear_eval.py` - Primary linear evaluation script with comprehensive features
- `linear_eval_v2.py` - Alternative implementation with additional utilities

Both scripts evaluate pre-trained SimCLR models on labeled OCT datasets by freezing the backbone network and training only a linear classification head.

## Features

- **Linear Evaluation Protocol**: Standard evaluation method for self-supervised learning
- **Multi-class Classification**: Support for any number of OCT disease classes
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and confusion matrices
- **Data Augmentation**: Adaptive histogram equalization and standard augmentations
- **Flexible Architecture**: Support for ResNet18 and ResNet50 backbones
- **DICOM Support**: Handle medical imaging formats (with pydicom)
- **Visualization**: Confusion matrix plotting and logging
- **Reproducible Results**: Seed setting for consistent experiments

## Requirements

```bash
pip install torch torchvision
pip install numpy tqdm pillow
pip install scikit-learn matplotlib
pip install pydicom  # Optional, for DICOM support
pip install pyyaml   # For configuration files
```

## Dataset Structure

Organize your labeled OCT dataset in the following structure:

```
data/
├── class1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── classN/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

Supported image formats: PNG, JPG, JPEG, BMP, TIF, TIFF, DCM

## Usage

### Basic Linear Evaluation

```bash
python linear_eval.py \
    --model-path /path/to/pretrained_simclr_model.pth \
    --data /path/to/labeled_oct_dataset \
    --n-classes 4 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.001
```

### Alternative Implementation

```bash
python linear_eval_v2.py \
    --checkpoint /path/to/pretrained_simclr_model.pth \
    --data-dir /path/to/labeled_oct_dataset \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0003
```

## Command Line Arguments

### Primary Script (`linear_eval.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | str | Required | Path to pretrained SimCLR model checkpoint |
| `--data` | str | Required | Path to labeled OCT dataset |
| `--n-classes` | int | Required | Number of classes in the dataset |
| `--dataset-name` | str | `oct_labeled` | Dataset name for logging |
| `--batch-size` | int | `128` | Batch size for training and evaluation |
| `--epochs` | int | `100` | Number of training epochs |
| `--lr` | float | `0.001` | Learning rate for linear classifier |
| `--weight-decay` | float | `1e-6` | Weight decay for optimizer |
| `--image-size` | int | `224` | Input image size |
| `--test-split` | float | `0.2` | Fraction of data for testing |
| `--arch` | str | `resnet50` | Backbone architecture (resnet18/resnet50) |
| `--out-dim` | int | `128` | Output dimension of SimCLR model |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--num-workers` | int | `4` | Number of data loading workers |
| `--gpu-index` | int | `0` | GPU index to use |

### Alternative Script (`linear_eval_v2.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | Required | Path to pretrained model checkpoint |
| `--data-dir` | str | Required | Path to OCT dataset directory |
| `--config` | str | `config.yml` | Path to configuration file |
| `--batch-size` | int | `64` | Batch size for training and evaluation |
| `--epochs` | int | `100` | Number of training epochs |
| `--lr` | float | `0.0003` | Learning rate |
| `--weight-decay` | float | `0.0008` | Weight decay |
| `--image-size` | int | `224` | Image size for training |
| `--test-split` | float | `0.2` | Proportion of data for testing |
| `--seed` | int | `42` | Random seed |
| `--arch` | str | `resnet18` | Model architecture |
| `--out-dim` | int | `128` | Output dimension of projection head |

## Key Components

### LinearClassifier Class

The `LinearClassifier` class wraps a pre-trained SimCLR model and adds a linear classification head:

```python
class LinearClassifier(nn.Module):
    def __init__(self, base_model, out_dim, n_classes):
        super(LinearClassifier, self).__init__()
        self.backbone = base_model
        
        # Freeze the backbone network
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Replace projection head with linear classifier
        self.fc = nn.Linear(dim_mlp, n_classes)
```

### LabeledOCTDataset Class

Custom dataset class for loading labeled OCT images:

- Automatically discovers class folders
- Supports multiple image formats including DICOM
- Handles grayscale conversion
- Provides class-to-index mappings

### Data Transformations

The evaluation uses different transformations for training and testing:

**Training Transforms:**
- Adaptive histogram equalization
- Random resized crop (scale 0.8-1.0)
- Random horizontal flip
- Normalization

**Test Transforms:**
- Adaptive histogram equalization  
- Resize and center crop
- Normalization

## Evaluation Metrics

The scripts compute comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision across classes
- **Recall**: Weighted average recall across classes
- **F1-Score**: Weighted average F1-score across classes
- **Confusion Matrix**: Detailed per-class performance

## Output Files

### Logs
- `linear_eval_{dataset_name}.log` - Training and evaluation logs
- `linear_eval_oct_{arch}.log` - Alternative logging format

### Model Checkpoints
- `linear_eval_{dataset_name}_best.pth` - Best model weights
- `linear_eval_best_{arch}.pth` - Best model with metadata

### Visualizations
- `confusion_matrix.png` - Confusion matrix heatmap

## Best Practices

### Hyperparameter Selection
- **Learning Rate**: Start with 0.001 for ResNet50, 0.0003 for ResNet18
- **Batch Size**: Use 64-128 depending on GPU memory
- **Epochs**: 100 epochs usually sufficient with early stopping
- **Weight Decay**: Use small values (1e-6 to 1e-4)

### Data Preparation
- Ensure balanced class distribution when possible
- Use consistent image preprocessing
- Verify DICOM files load correctly if using medical data

### Reproducibility
- Set random seeds for consistent results
- Use the same train/test split across experiments
- Document hyperparameters and data splits

## Example Workflow

1. **Train SimCLR Model**: First train a SimCLR model on unlabeled OCT data
2. **Prepare Labeled Data**: Organize labeled OCT images by class
3. **Run Linear Evaluation**: Use the scripts to evaluate learned representations
4. **Analyze Results**: Review metrics and confusion matrices

```bash
# Example complete workflow
python linear_eval.py \
    --model-path simclr_oct_model.pth \
    --data ./oct_labeled_dataset \
    --n-classes 4 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --arch resnet50 \
    --seed 42
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Import Errors**: Ensure all dependencies are installed
3. **DICOM Loading**: Install pydicom for medical image support
4. **Class Imbalance**: Consider weighted loss functions for imbalanced datasets

### Performance Tips

- Use GPU acceleration when available
- Adjust number of workers based on CPU cores
- Monitor GPU memory usage during training
- Use mixed precision training for larger models

## Dependencies

The code requires the original SimCLR implementation files:
- `main.py` - Contains ResNetOCTSimCLR, OCTImageDataset, AdaptiveEqualization
- Ensure these classes are properly imported

## License

Please ensure compliance with relevant licenses for medical imaging data and deep learning frameworks.

## Citation

If you use this code in your research, please cite the original SimCLR paper and any relevant OCT dataset papers.

---

For additional questions or issues, please refer to the logging output and ensure all dependencies are properly installed.
