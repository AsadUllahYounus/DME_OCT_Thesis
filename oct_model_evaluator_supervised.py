import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model architecture (same as training script)
class ResNetOCTClassifier(nn.Module):
    def __init__(self, base_model, num_classes, pretrained=False):
        super(ResNetOCTClassifier, self).__init__()
        
        self.resnet_dict = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50
        }

        self.backbone = self._get_basemodel(base_model, pretrained)
        
        # Modify the first layer to accept grayscale images (1 channel instead of 3)
        if pretrained:
            # If using pretrained weights, we need to average the weights across RGB channels
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Average the pretrained RGB weights to work with grayscale
            with torch.no_grad():
                self.backbone.conv1.weight[:, 0, :, :] = old_conv.weight.mean(1)
        else:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer for classification
        dim_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(dim_features, num_classes)

    def _get_basemodel(self, model_name, pretrained):
        try:
            if pretrained:
                model = self.resnet_dict[model_name](weights='IMAGENET1K_V1')
            else:
                model = self.resnet_dict[model_name](weights=None)
        except KeyError:
            raise ValueError(f"Invalid backbone architecture. Choose from: {list(self.resnet_dict.keys())}")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

# Test dataset class
class TestOCTDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_to_idx=None):
        """
        Args:
            data_dir (string): Directory with subdirectories for each class
            transform (callable, optional): Optional transform to be applied on a sample.
            class_to_idx (dict): Mapping from class names to indices
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get class names from subdirectories
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        
        if class_to_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Get all image file paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.dcm')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(class_idx)
        
        print(f"Found {len(self.image_paths)} test images across {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
        
        # Print class distribution
        class_counts = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Test set class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Handle DICOM files if present
            if img_path.lower().endswith('.dcm'):
                try:
                    import pydicom
                    ds = pydicom.dcmread(img_path)
                    image = ds.pixel_array
                    # Normalize to 0-255
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                except ImportError:
                    print("Warning: pydicom not installed. DICOM files will be skipped.")
                    # Return a blank grayscale image as a fallback
                    image = Image.new('L', (512, 512), 128)
            else:
                # Open regular image files
                image = Image.open(img_path)
                
                # Convert to grayscale if not already
                if image.mode != 'L':
                    image = ImageOps.grayscale(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank grayscale image as a fallback
            image = Image.new('L', (512, 512), 128)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_test_transform(image_size):
    """Get test transforms without augmentations"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def load_model(checkpoint_path, num_classes, arch, device):
    """Load a saved model"""
    model = ResNetOCTClassifier(base_model=arch, num_classes=num_classes)
    
    # Load checkpoint
    if device.type == 'cpu':
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    
    return model

def calculate_sensitivity_specificity(y_true, y_pred, num_classes):
    """Calculate sensitivity and specificity for each class"""
    cm = confusion_matrix(y_true, y_pred)
    
    sensitivity = []
    specificity = []
    
    for i in range(num_classes):
        # True Positives, False Negatives, False Positives, True Negatives
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity.append(sens)
        
        # Specificity = TN / (TN + FP)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    
    return sensitivity, specificity

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def evaluate_model(model, test_loader, device, class_names, save_dir=None):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    # Calculate sensitivity and specificity
    sensitivity, specificity = calculate_sensitivity_specificity(
        all_labels, all_predictions, len(class_names)
    )
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Macro Sensitivity: {macro_sensitivity:.4f}")
    print(f"Macro Specificity: {macro_specificity:.4f}")
    
    print("\n" + "-"*60)
    print("PER-CLASS METRICS")
    print("-"*60)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Sensitivity':<12} {'Specificity':<12} {'Support':<8}")
    print("-"*60)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} "
              f"{sensitivity[i]:<12.4f} {specificity[i]:<12.4f} {support[i]:<8}")
    
    # Generate and save detailed report
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save confusion matrix
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_labels, all_predictions, class_names, cm_path)
        
        # Save detailed classification report
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        report_path = os.path.join(save_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Macro Precision: {macro_precision:.4f}\n")
            f.write(f"Macro Recall: {macro_recall:.4f}\n")
            f.write(f"Macro F1-Score: {macro_f1:.4f}\n")
            f.write(f"Macro Sensitivity: {macro_sensitivity:.4f}\n")
            f.write(f"Macro Specificity: {macro_specificity:.4f}\n\n")
            f.write(report)
        
        print(f"\nDetailed results saved to {save_dir}")
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Support': support
        })
        
        csv_path = os.path.join(save_dir, 'metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'support': support,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained OCT classification model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved model checkpoint (.pth.tar file)')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test dataset directory')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Model architecture (default: resnet18)')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Directory to save evaluation results (optional)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Check if test data directory exists
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data directory not found: {args.test_data}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, args.num_classes, args.arch, device)
    
    # Prepare test dataset
    print("Preparing test dataset...")
    transform = get_test_transform(args.image_size)
    test_dataset = TestOCTDataset(args.test_data, transform=transform)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True
    )
    
    # Evaluate model
    results = evaluate_model(
        model, test_loader, device, 
        test_dataset.classes, args.save_results
    )
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
