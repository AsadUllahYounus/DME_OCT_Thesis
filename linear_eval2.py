import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image, ImageOps

# Import necessary components from the original SimCLR implementation
# Make sure these files are in the same directory
from main import ResNetOCTSimCLR, OCTImageDataset, AdaptiveEqualization, InvalidBackboneError

# Import the LabeledOCTDataset from the second file
from linear_eval import LabeledOCTDataset, get_transform

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

def get_oct_data_loaders(data_dir, image_size, batch_size=64, test_split=0.2, seed=42):
    """
    Create data loaders for OCT images
    """
    # Create train and test transforms
    train_transform = get_transform(image_size, is_train=True)
    test_transform = get_transform(image_size, is_train=False)
    
    # Load the full dataset
    full_dataset = LabeledOCTDataset(data_dir, transform=train_transform)
    
    # Split into train and test sets
    test_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Update test transform
    test_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, test_loader, full_dataset.class_to_idx

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

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on the test set"""
    model.eval()
    top1_accuracy = 0
    test_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc="Evaluating")):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            test_loss += loss.item()
            
            # Get predictions
            _, pred = torch.max(logits, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
            top1, = accuracy(logits, y_batch, topk=(1,))
            top1_accuracy += top1[0]
    
    # Calculate average metrics
    top1_accuracy /= (counter + 1)
    test_loss /= (counter + 1)
    
    # Calculate additional metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'top1_accuracy': top1_accuracy.item(),
        'loss': test_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Setup argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Linear Evaluation for OCT SimCLR')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the pre-trained model checkpoint')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to the configuration file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the OCT dataset directory with class subfolders')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0008,
                        help='Weight decay')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Model architecture')
    parser.add_argument('--out-dim', type=int, default=128,
                        help='Output dimension of the projection head')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load the configuration file if it exists
    config = args.__dict__
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config.update(yaml.safe_load(file))
    
    # Load the dataset
    train_loader, test_loader, class_to_idx = get_oct_data_loaders(
        args.data_dir, args.image_size, args.batch_size, args.test_split, args.seed
    )
    
    # Number of classes
    num_classes = len(class_to_idx)
    print(f"Dataset loaded with {num_classes} classes")
    
    # Initialize the model
    simclr_model = ResNetOCTSimCLR(base_model=args.arch, out_dim=args.out_dim)
    
    # Load pre-trained weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # Load the pre-trained model
    simclr_model.load_state_dict(state_dict)
    
    # Create model for linear evaluation
    model = torch.nn.Sequential(
        simclr_model,
        torch.nn.Linear(args.out_dim, num_classes)
    ).to(device)
    
    # Freeze all layers except the last linear layer
    for name, param in model.named_parameters():
        if name not in ['1.weight', '1.bias']:
            param.requires_grad = False
    
    # Check that only the linear layer is trainable
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Set up logging
    logging.basicConfig(
        filename=f'linear_eval_oct_{args.arch}.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        top1_train_accuracy = 0
        train_loss = 0
        
        for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            train_loss += loss.item()
            
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate average training metrics
        top1_train_accuracy /= (counter + 1)
        train_loss /= (counter + 1)
        
        # Evaluation phase
        eval_metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Log results
        log_message = (
            f"Epoch {epoch+1}/{args.epochs}\n"
            f"Train Loss: {train_loss:.4f}, Train Acc: {top1_train_accuracy.item():.2f}%\n"
            f"Test Loss: {eval_metrics['loss']:.4f}, Test Acc: {eval_metrics['top1_accuracy']:.2f}%, "
            f"Top1 Acc: {eval_metrics['top1_accuracy']:.2f}%\n"
            f"Precision: {eval_metrics['precision']:.4f}, Recall: {eval_metrics['recall']:.4f}, "
            f"F1: {eval_metrics['f1']:.4f}"
        )
        
        print(log_message)
        logging.info(log_message)
        
        # Save best model
        if eval_metrics['top1_accuracy'] > best_acc:
            best_acc = eval_metrics['top1_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'class_to_idx': class_to_idx
            }, f'linear_eval_best_{args.arch}.pth')
            logging.info(f"New best model saved with accuracy: {best_acc:.2f}%")
    
    # Final evaluation with best model
    checkpoint = torch.load(f'linear_eval_best_{args.arch}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = evaluate_model(model, test_loader, criterion, device)
    
    # Reverse the class_to_idx mapping for plotting
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    # Plot confusion matrix
    plot_confusion_matrix(final_metrics['confusion_matrix'], class_names)
    
    # Print final results
    print("\nFinal Evaluation Results:")
    print(f"Top-1 Accuracy: {final_metrics['top1_accuracy']:.2f}%")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"Confusion Matrix saved to confusion_matrix.png")
    
    # Log final results
    logging.info("\nFinal Evaluation Results:")
    logging.info(f"Top-1 Accuracy: {final_metrics['top1_accuracy']:.2f}%")
    logging.info(f"Precision: {final_metrics['precision']:.4f}")
    logging.info(f"Recall: {final_metrics['recall']:.4f}")
    logging.info(f"F1 Score: {final_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
