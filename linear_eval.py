import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import necessary classes from original SimCLR implementation
# Assumes the original file is named simclr_oct.py
from main import ResNetOCTSimCLR, OCTImageDataset, InvalidBackboneError, AdaptiveEqualization

class LinearClassifier(nn.Module):
    """Linear classifier head for evaluating learned representations"""
    def __init__(self, base_model, out_dim, n_classes):
        super(LinearClassifier, self).__init__()
        self.backbone = base_model
       
        # Freeze the backbone network
        for param in self.backbone.parameters():
            param.requires_grad = False
           
        # Replace the projection head with a linear classifier
        dim_mlp = self.backbone.backbone.fc[0].in_features
        self.backbone.backbone.fc = nn.Identity()
       
        # Add linear classifier
        self.fc = nn.Linear(dim_mlp, n_classes)
       
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

class LabeledOCTDataset(Dataset):
    """Dataset class for labeled OCT images"""
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Args:
            data_dir (string): Directory with all OCT images organized in class subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string): 'train' or 'test' split.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
       
        # Get all image files and their corresponding labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
       
        # Find class folders (each subfolder represents a class)
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        class_dirs.sort()  # Sort for reproducibility
       
        # Create class to index mapping
        for idx, class_name in enumerate(class_dirs):
            self.class_to_idx[class_name] = idx
           
        # Get all images with their labels
        for class_name in class_dirs:
            class_path = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
           
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.dcm')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(class_idx)
       
        # Sort for reproducibility
        paired_data = list(zip(self.image_paths, self.labels))
        paired_data.sort(key=lambda x: x[0])
        self.image_paths, self.labels = zip(*paired_data)
       
        print(f"Found {len(self.image_paths)} labeled OCT images in {len(class_dirs)} classes")
        for class_name, idx in self.class_to_idx.items():
            count = self.labels.count(idx)
            print(f"  Class '{class_name}': {count} images")

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
                    image = Image.new('L', (512, 512), 128)
            else:
                # Open regular image files
                image = Image.open(img_path)
               
                # Convert to grayscale if not already
                if image.mode != 'L':
                    image = ImageOps.grayscale(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('L', (512, 512), 128)
       
        if self.transform:
            image = self.transform(image)
           
        return image, label

def get_transform(image_size, is_train=True):
    """Get the transformations for evaluation"""
    if is_train:
        return transforms.Compose([
            AdaptiveEqualization(),
            transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        return transforms.Compose([
            AdaptiveEqualization(),
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

def evaluate(model, data_loader, device):
    """Evaluate the linear classifier"""
    model.eval()
    all_preds = []
    all_labels = []
   
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
           
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
           
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
   
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
   
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def train_linear_classifier(model, train_loader, test_loader, device, args):
    """Train and evaluate the linear classifier"""
    # Set up the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
   
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
   
    # Loss function
    criterion = nn.CrossEntropyLoss()
   
    # Set up logging
    logging.basicConfig(filename=f'linear_eval_{args.dataset_name}.log', level=logging.INFO)
   
    best_acc = 0.0
   
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
       
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
           
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
           
            progress_bar.set_postfix({
                'loss': train_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
       
        # Evaluation phase
        eval_metrics = evaluate(model, test_loader, device)
       
        # Log results
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        logging.info(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {100.*correct/total:.2f}%")
        logging.info(f"Test Acc: {eval_metrics['accuracy']*100:.2f}% | F1: {eval_metrics['f1']:.4f}")
       
        # Update learning rate based on validation accuracy
        scheduler.step(eval_metrics['accuracy'])
       
        # Save best model
        if eval_metrics['accuracy'] > best_acc:
            best_acc = eval_metrics['accuracy']
            torch.save(model.state_dict(), f'linear_eval_{args.dataset_name}_best.pth')
            logging.info(f"Best model saved with accuracy: {best_acc*100:.2f}%")
   
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'linear_eval_{args.dataset_name}_best.pth'))
    final_metrics = evaluate(model, test_loader, device)
   
    logging.info("Final Evaluation Metrics:")
    logging.info(f"Accuracy: {final_metrics['accuracy']*100:.2f}%")
    logging.info(f"Precision: {final_metrics['precision']:.4f}")
    logging.info(f"Recall: {final_metrics['recall']:.4f}")
    logging.info(f"F1 Score: {final_metrics['f1']:.4f}")
    logging.info(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")
   
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description='Linear Evaluation for SimCLR OCT')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the pretrained SimCLR model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to labeled OCT dataset (with class subfolders)')
    parser.add_argument('--dataset-name', default='oct_labeled',
                        help='Dataset name for logging')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for linear evaluation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for linear classifier')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='Weight decay for optimizer')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for training and evaluation')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50'],
                        help='Base model architecture')
    parser.add_argument('--out-dim', type=int, default=128,
                        help='Output dimension of SimCLR model')
    parser.add_argument('--n-classes', type=int, required=True,
                        help='Number of classes in the dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--gpu-index', type=int, default=0,
                        help='GPU index to use')
   
    args = parser.parse_args()
   
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
   
    # Set device
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
   
    # Load the dataset
    train_transform = get_transform(args.image_size, is_train=True)
    test_transform = get_transform(args.image_size, is_train=False)
   
    full_dataset = LabeledOCTDataset(args.data, transform=train_transform)
   
    # Split into train and test sets
    test_size = int(args.test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
   
    # Update test transform
    test_dataset.dataset.transform = test_transform
   
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
   
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
   
    # Initialize the SimCLR model
    simclr_model = ResNetOCTSimCLR(base_model=args.arch, out_dim=args.out_dim)
   
    # Load pre-trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    simclr_model.load_state_dict(checkpoint['state_dict'])
   
    # Create linear classifier
    model = LinearClassifier(simclr_model, args.out_dim, args.n_classes).to(device)
   
    # Train and evaluate
    print("Starting linear evaluation...")
    metrics = train_linear_classifier(model, train_loader, test_loader, device, args)
   
    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

if __name__ == "__main__":
    main()
