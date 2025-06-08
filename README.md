# SimCLR for OCT Images

A PyTorch implementation of enhanced SimCLR (Simple Framework for Contrastive Learning of Visual Representations) specifically optimized for Optical Coherence Tomography (OCT) medical images.

## Overview

This project adapts the SimCLR self-supervised learning framework for OCT image analysis. OCT images are commonly used in ophthalmology and require specialized preprocessing and augmentation techniques due to their grayscale nature and unique structural characteristics.

## Features

- **OCT-Optimized Data Augmentations**: Custom transformations designed for medical imaging
- **Grayscale Support**: Modified ResNet architecture for single-channel OCT images
- **Adaptive Histogram Equalization**: Enhanced contrast processing for medical images
- **Custom Gaussian Blur**: OCT-specific blur augmentation
- **DICOM Support**: Handle medical DICOM format files
- **Mixed Precision Training**: FP16 support for faster training
- **Comprehensive Logging**: TensorBoard integration and detailed logging

## Requirements

```bash
torch>=1.8.0
torchvision>=0.9.0
numpy
Pillow
PyYAML
tqdm
tensorboard
pydicom  # Optional, for DICOM file support
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd simclr-oct
```

2. Install dependencies:
```bash
pip install torch torchvision numpy Pillow PyYAML tqdm tensorboard
# Optional: for DICOM support
pip install pydicom
```

## Dataset Structure

Organize your OCT images in the following structure:
```
datasets/
└── oct_images/
    ├── image1.png
    ├── image2.jpg
    ├── image3.tiff
    ├── scan1.dcm  # DICOM files supported
    └── ...
```

Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF, DCM

## Usage

### Basic Training

```bash
python main.py \
    --data ./datasets/oct_images \
    --dataset-name oct \
    --arch resnet18 \
    --epochs 200 \
    --batch-size 64 \
    --image-size 224
```

### Advanced Training Options

```bash
python main.py \
    --data ./datasets/oct_images \
    --dataset-name oct \
    --arch resnet50 \
    --epochs 500 \
    --batch-size 32 \
    --lr 0.0003 \
    --temperature 0.07 \
    --fp16-precision \
    --gpu-index 0 \
    --image-size 512
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `./datasets` | Path to dataset directory |
| `--dataset-name` | `oct` | Dataset type (oct, cifar10, stl10) |
| `--arch` | `resnet18` | Model architecture (resnet18, resnet50) |
| `--epochs` | `200` | Number of training epochs |
| `--batch-size` | `64` | Mini-batch size |
| `--lr` | `0.0003` | Initial learning rate |
| `--temperature` | `0.07` | Temperature parameter for contrastive loss |
| `--out_dim` | `128` | Output dimension of projection head |
| `--image-size` | `224` | Input image size |
| `--fp16-precision` | `False` | Enable mixed precision training |
| `--gpu-index` | `0` | GPU device index |

## Model Architecture

The model uses a modified ResNet backbone with the following adaptations for OCT images:

1. **Single Channel Input**: First convolutional layer modified to accept grayscale images
2. **Projection Head**: MLP head for contrastive learning
3. **No Pretrained Weights**: Training from scratch on medical data

```python
# Key modifications:
self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.backbone.fc = nn.Sequential(
    nn.Linear(dim_mlp, dim_mlp), 
    nn.ReLU(), 
    nn.Linear(dim_mlp, out_dim)
)
```

## Data Augmentations

OCT-specific augmentation pipeline:

1. **Adaptive Equalization**: Enhances contrast in medical images
2. **Conservative Cropping**: `scale=(0.5, 1.0)` to preserve anatomical structures
3. **Limited Rotation**: `±10°` to maintain medical relevance
4. **Gentle Color Jitter**: Reduced brightness/contrast changes
5. **Gaussian Blur**: Custom implementation for single-channel images

## Output Files

Training generates the following outputs:

- `runs/`: TensorBoard logs and training metrics
- `checkpoint_XXXX.pth.tar`: Model checkpoints
- `config.yml`: Training configuration
- `training.log`: Detailed training logs

## Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir runs/
```

Metrics tracked:
- Contrastive loss
- Top-1 and Top-5 accuracy
- Learning rate schedule

## Pretrained Models

After training, use the learned representations for downstream tasks:

```python
# Load pretrained SimCLR model
checkpoint = torch.load('checkpoint_0200.pth.tar')
model = ResNetOCTSimCLR(base_model='resnet18', out_dim=128)
model.load_state_dict(checkpoint['state_dict'])

# Extract features
features = model.backbone[:-1](images)  # Remove projection head
```

## Performance Tips

1. **Batch Size**: Larger batches (64-128) typically work better for contrastive learning
2. **Learning Rate**: Start with 0.0003, scale with batch size if needed
3. **Temperature**: 0.07 works well for most cases, lower values (0.05) for harder negatives
4. **Image Size**: Balance between quality and computational cost (224-512px)
5. **Epochs**: OCT images may need 300-500 epochs for convergence

## Common Issues

### CUDA Out of Memory
- Reduce `--batch-size`
- Enable `--fp16-precision`
- Use smaller `--image-size`

### Slow Training
- Increase `--workers` for data loading
- Enable `--fp16-precision`
- Use multiple GPUs (modify code for DataParallel)

### Poor Convergence
- Increase `--epochs`
- Adjust `--temperature` (try 0.05-0.1)
- Check data quality and preprocessing

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={International conference on machine learning},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Original SimCLR implementation by Google Research
- PyTorch team for the framework
- Medical imaging community for OCT datasets
