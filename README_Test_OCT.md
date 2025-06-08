# OCT Model Test Evaluation Documentation

## Overview

This test_oct_model script provides comprehensive evaluation capabilities for trained OCT (Optical Coherence Tomography) classification models. It loads a trained model checkpoint and evaluates its performance on a test dataset, generating detailed metrics, visualizations, and error analysis.

## Features

- **Model Loading**: Automatically loads trained SimCLR-based OCT classification models
- **Test Set Evaluation**: Processes test images organized in class subdirectories
- **Performance Metrics**: Calculates accuracy, precision, recall, and F1-score
- **Confusion Matrix**: Generates and saves confusion matrix visualization
- **Error Analysis**: Identifies and visualizes misclassified examples
- **Detailed Reporting**: Exports comprehensive results to CSV format

## Requirements

### Dependencies

```python
torch
numpy
PIL (Pillow)
matplotlib
torchvision
scikit-learn
seaborn
pandas
```

### Required Modules

- `main.py`: Contains the `ResNetOCTSimCLR` model class
- `linear_eval.py`: Contains the `get_transform` function for image preprocessing

## Usage

### Basic Command

```bash
python test_eval.py --checkpoint path/to/model.pth --test-dir path/to/test/directory
```

### Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--checkpoint` | str | Yes | - | Path to the trained model checkpoint file |
| `--test-dir` | str | Yes | - | Path to test directory with class subdirectories |
| `--image-size` | int | No | 224 | Image size for preprocessing (width × height) |
| `--output` | str | No | test_results.csv | Path to save detailed evaluation results |

### Example Usage

```bash
# Basic evaluation
python test_eval.py --checkpoint models/best_model_resnet18.pth --test-dir data/test/

# Custom image size and output file
python test_oct_model.py \
    --checkpoint models/best_model_resnet50.pth \
    --test-dir data/test/ \
    --image-size 256 \
    --output evaluation_results.csv
```

## Test Directory Structure

The test directory should be organized with class subdirectories:

```
test_dir/
├── class1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── class3/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

## Functions Documentation

### `load_model(checkpoint_path, device)`

Loads a trained OCT classification model from a checkpoint file.

**Parameters:**
- `checkpoint_path` (str): Path to the model checkpoint
- `device` (str): Device to load the model on ('cuda' or 'cpu')

**Returns:**
- `model`: Loaded PyTorch model
- `class_to_idx` (dict): Mapping from class names to indices
- `idx_to_class` (dict): Mapping from indices to class names

### `evaluate_test_set(model, test_dir, transform, class_to_idx, device)`

Evaluates the model on all images in the test directory.

**Parameters:**
- `model`: Trained PyTorch model
- `test_dir` (str): Path to test directory
- `transform`: Image preprocessing transform
- `class_to_idx` (dict): Class name to index mapping
- `device` (str): Device for inference

**Returns:**
- `all_predictions` (list): Model predictions for all images
- `all_labels` (list): True labels for all images
- `all_confidences` (list): Prediction confidences
- `all_file_paths` (list): Paths to all processed images

### `plot_confusion_matrix(cm, class_names, title)`

Creates and saves a confusion matrix visualization.

**Parameters:**
- `cm` (array): Confusion matrix from sklearn
- `class_names` (list): List of class names for labeling
- `title` (str): Title for the plot

**Output:** Saves `test_confusion_matrix.png`

### `plot_misclassified_examples(misclassified, idx_to_class, max_examples)`

Visualizes examples of misclassified images.

**Parameters:**
- `misclassified` (list): List of misclassified image data
- `idx_to_class` (dict): Index to class name mapping
- `max_examples` (int): Maximum number of examples to show

**Output:** Saves `misclassified_examples.png`

## Output Files

### 1. Console Output

The script prints comprehensive evaluation metrics:

```
Test Set Evaluation Results:
Number of test images: 1000
Accuracy: 0.8500
Precision: 0.8475
Recall: 0.8500
F1 Score: 0.8487
Total misclassified: 150 of 1000 (15.0%)
```

### 2. Confusion Matrix (`test_confusion_matrix.png`)

A heatmap visualization showing the confusion matrix with:
- True labels on the y-axis
- Predicted labels on the x-axis
- Color-coded cell values showing prediction counts

### 3. Misclassified Examples (`misclassified_examples.png`)

Visual examples of incorrectly classified images showing:
- Original image
- True class label
- Predicted class label
- Prediction confidence

### 4. Detailed Results CSV

A comprehensive CSV file containing:

| Column | Description |
|--------|-------------|
| `image_path` | Full path to the image file |
| `true_class` | Actual class label |
| `predicted_class` | Model's predicted class |
| `confidence` | Prediction confidence score |
| `correct` | Boolean indicating if prediction was correct |

## Error Handling

- **Invalid Images**: Skips files that cannot be loaded or processed
- **Missing Classes**: Warns about classes in test data not found in model
- **Empty Directories**: Handles empty test directories gracefully
- **File Format**: Only processes common image formats (.png, .jpg, .jpeg)

## Performance Considerations

- **GPU Usage**: Automatically detects and uses CUDA if available
- **Memory Management**: Processes images individually to avoid memory issues
- **Batch Processing**: Uses single image inference for detailed per-image analysis

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure checkpoint file exists and is compatible
   - Check that required model classes are imported

2. **Class Mismatch**
   - Verify test directory class names match training classes
   - Check for typos in directory names

3. **Image Processing Errors**
   - Ensure images are in supported formats
   - Check file permissions and corruption

4. **Memory Issues**
   - Reduce image size with `--image-size` parameter
   - Ensure sufficient GPU/CPU memory

### Example Error Messages

```
Warning: Class 'unknown_class' not found in model classes. Skipping.
Error processing /path/to/image.jpg: Cannot identify image file
```

## Integration Notes

This evaluation script is designed to work with:
- SimCLR-based OCT classification models
- Standard PyTorch checkpoint format
- Scikit-learn metrics and visualization tools

For custom model architectures, modify the `load_model()` function accordingly.
