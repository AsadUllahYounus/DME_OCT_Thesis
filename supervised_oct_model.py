import os
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import argparse
from tqdm import tqdm
import logging
import sys
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Define exceptions first to avoid circular imports
class BaseModelException(Exception):
    """Base exception"""

class InvalidBackboneError(BaseModelException):
    """Raised when the choice of backbone Convnet is invalid."""

class InvalidDatasetSelection(BaseModelException):
    """Raised when the choice of dataset is invalid."""

# Utility functions
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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
    plt.savefig(save_path)
    plt.close()

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Modified GaussianBlur for grayscale OCT images (from original SimCLR code)
class GaussianBlur(object):
    """Blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(1, 1, kernel_size=(kernel_size, 1),
                               stride=1, padding=0, bias=False, groups=1)
        self.blur_v = nn.Conv2d(1, 1, kernel_size=(1, kernel_size),
                               stride=1, padding=0, bias=False, groups=1)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        # Convert the PIL image to tensor, apply blur, and convert back
        img = self.pil_to_tensor(img).unsqueeze(0)
        img = self.blur(img).squeeze(0)
        img = self.tensor_to_pil(img)
        return img

# Supervised OCT dataset class
class SupervisedOCTDataset(Dataset):
    def __init__(self, image_paths, labels, class_names, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels (integers)
            class_names (list): List of class names
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        
        print(f"Dataset created with {len(self.image_paths)} images across {len(set(labels))} classes")
        
        # Print class distribution
        class_counts = {}
        for label in labels:
            class_name = class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Class distribution:")
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

class OCTDatasetLoader:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self._load_dataset()

    def _load_dataset(self):
        """Load all images from class subdirectories"""
        # Get class names from subdirectories
        self.class_names = sorted([d for d in os.listdir(self.root_folder) 
                                 if os.path.isdir(os.path.join(self.root_folder, d))])
        
        if len(self.class_names) == 0:
            raise ValueError(f"No class directories found in {self.root_folder}")
        
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Get all image file paths and labels
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_folder, class_name)
            class_idx = class_to_idx[class_name]
            
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.dcm')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(class_idx)
        
        print(f"Found {len(self.image_paths)} images across {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")

    @staticmethod
    def get_oct_pipeline_transform(size, is_training=True):
        """Return basic transformations for OCT images."""
        if is_training:
            # Training transformations with basic augmentation
            data_transforms = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalized for grayscale
            ])
        else:
            # Validation/test transformations - no augmentation
            data_transforms = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        return data_transforms

    def get_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """Split the dataset into train, validation, and test sets"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        # First split: train vs (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            self.image_paths, self.labels, 
            test_size=(val_ratio + test_ratio), 
            random_state=random_state, 
            stratify=self.labels
        )
        
        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels,
                test_size=test_ratio/(val_ratio + test_ratio),
                random_state=random_state,
                stratify=temp_labels
            )
        elif val_ratio > 0:
            val_paths, val_labels = temp_paths, temp_labels
            test_paths, test_labels = [], []
        else:
            test_paths, test_labels = temp_paths, temp_labels
            val_paths, val_labels = [], []
        
        return {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels),
            'test': (test_paths, test_labels)
        }

    def get_dataset(self, image_paths, labels, image_size=224, is_training=True):
        """Create a dataset from given paths and labels"""
        transform = self.get_oct_pipeline_transform(image_size, is_training)
        dataset = SupervisedOCTDataset(image_paths, labels, self.class_names, transform=transform)
        return dataset

# Model architecture for supervised learning
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
            raise InvalidBackboneError(
                f"Invalid backbone architecture. Choose from: {list(self.resnet_dict.keys())}")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

# Main training class
class SupervisedTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        
        # Use class weights if specified
        if hasattr(self.args, 'class_weights') and self.args.class_weights is not None:
            weights = torch.tensor(self.args.class_weights, dtype=torch.float32).to(self.args.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        self.best_acc = 0.0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        scaler = GradScaler(enabled=self.args.fp16_precision)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.args.fp16_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log to tensorboard
            if batch_idx % self.args.log_every_n_steps == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/accuracy', 100.*correct/total, global_step)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def validate(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions and labels for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', epoch_loss, epoch)
        self.writer.add_scalar('val/accuracy', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc, all_predictions, all_labels

    def train(self, train_loader, val_loader=None, class_names=None):
        # Save config file
        save_config_file(self.writer.log_dir, self.args)
        
        logging.info(f"Start supervised training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        for epoch in range(self.args.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc, val_predictions, val_labels = self.validate(val_loader, epoch)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Save best model
                is_best = val_acc > self.best_acc
                if is_best:
                    self.best_acc = val_acc
                    
                    # Generate detailed validation report
                    if class_names is None:
                        class_names = [f'Class_{i}' for i in range(len(set(val_labels)))]
                    
                    report = classification_report(val_labels, val_predictions, 
                                                 target_names=class_names, output_dict=True)
                    
                    # Save classification report
                    report_path = os.path.join(self.writer.log_dir, 'classification_report.txt')
                    with open(report_path, 'w') as f:
                        f.write(classification_report(val_labels, val_predictions, target_names=class_names))
                    
                    # Save confusion matrix
                    cm_path = os.path.join(self.writer.log_dir, 'confusion_matrix.png')
                    plot_confusion_matrix(val_labels, val_predictions, class_names, cm_path)
                
                logging.info(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            else:
                is_best = train_acc > self.best_acc
                if is_best:
                    self.best_acc = train_acc
                logging.info(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # Save checkpoint
            checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            }, is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            
            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader else train_loss)
                else:
                    self.scheduler.step()
        
        logging.info("Training has finished.")
        logging.info(f"Best accuracy: {self.best_acc:.2f}%")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

# Argument parsing and main function
def parse_args():
    model_names = ['resnet18', 'resnet50']
    
    parser = argparse.ArgumentParser(description='PyTorch Supervised Learning for OCT Images')
    parser.add_argument('-data', metavar='DIR', default='./datasets',
                        help='path to dataset (should contain class subdirectories)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--image-size', default=224, type=int, help='Image size for OCT dataset.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained ImageNet weights')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--train-ratio', default=0.9, type=float,
                        help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--val-ratio', default=0.05, type=float,
                        help='Ratio of data to use for validation (default: 0.15)')
    parser.add_argument('--test-ratio', default=0.05, type=float,
                        help='Ratio of data to use for testing (default: 0.15)')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Load and split the dataset
    dataset_loader = OCTDatasetLoader(args.data)
    splits = dataset_loader.get_splits(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    
    # Auto-detect number of classes
    num_classes = len(dataset_loader.class_names)
    print(f"Auto-detected {num_classes} classes: {dataset_loader.class_names}")
    
    # Create datasets
    train_paths, train_labels = splits['train']
    train_dataset = dataset_loader.get_dataset(train_paths, train_labels, args.image_size, is_training=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    
    # Create validation loader if validation data exists
    val_loader = None
    val_paths, val_labels = splits['val']
    if len(val_paths) > 0:
        val_dataset = dataset_loader.get_dataset(val_paths, val_labels, args.image_size, is_training=False)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        print(f"Validation dataset created with {len(val_dataset)} images")
    else:
        print("No validation dataset created.")

    # Create model
    model = ResNetOCTClassifier(
        base_model=args.arch, 
        num_classes=num_classes, 
        pretrained=args.pretrained
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    # Train the model
    trainer = SupervisedTrainer(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    trainer.train(train_loader, val_loader, class_names=dataset_loader.class_names)

if __name__ == "__main__":
    main()
