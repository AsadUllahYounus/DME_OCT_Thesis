import torch
import numpy as np
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import pandas as pd

# Import necessary components
from main import ResNetOCTSimCLR
from linear_eval import get_transform

def load_model(checkpoint_path, device):
    """Load the trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    arch = checkpoint_path.split('_')[-1].split('.')[0]  # Extract architecture
    out_dim = 128  # Default
    num_classes = len(checkpoint['class_to_idx'])
    class_to_idx = checkpoint['class_to_idx']
    
    # Create model
    simclr_model = ResNetOCTSimCLR(base_model=arch, out_dim=out_dim)
    
    model = torch.nn.Sequential(
        simclr_model,
        torch.nn.Linear(out_dim, num_classes)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    return model, class_to_idx, idx_to_class

def evaluate_test_set(model, test_dir, transform, class_to_idx, device):
    """Evaluate model on a test set with class subdirectories"""
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_file_paths = []
    
    # Process each class directory
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
            
        # Skip if not in our class mapping
        if class_name not in class_to_idx:
            print(f"Warning: Class '{class_name}' not found in model classes. Skipping.")
            continue
            
        class_idx = class_to_idx[class_name]
        
        # Process each image in the class directory
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg','.bmp', '.tif', '.tiff', '.dcm')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load and transform image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, prediction = torch.max(probabilities, 1)
                
                # Record results
                all_predictions.append(prediction.item())
                all_labels.append(class_idx)
                all_confidences.append(confidence.item())
                all_file_paths.append(img_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return all_predictions, all_labels, all_confidences, all_file_paths

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plot confusion matrix with seaborn for better visualization"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png')
    plt.close()
    
def plot_misclassified_examples(misclassified, idx_to_class, max_examples=10):
    """Plot examples of misclassified images"""
    n = min(len(misclassified), max_examples)
    if n == 0:
        return
        
    fig, axes = plt.subplots(n, 1, figsize=(12, 4*n))
    if n == 1:
        axes = [axes]
        
    for i in range(n):
        img_path, true_label, pred_label, conf = misclassified[i]
        img = Image.open(img_path)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {idx_to_class[true_label]} | Predicted: {idx_to_class[pred_label]} (Conf: {conf:.2f})")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate OCT model on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--test-dir', type=str, required=True,
                      help='Path to test directory with class subdirectories')
    parser.add_argument('--image-size', type=int, default=224,
                      help='Image size for testing')
    parser.add_argument('--output', type=str, default='test_results.csv',
                      help='Path to save detailed results')
    
    args = parser.parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model, class_to_idx, idx_to_class = load_model(args.checkpoint, device)
    print(f"Model loaded with classes: {list(idx_to_class.values())}")
    
    # Get transform
    transform = get_transform(args.image_size, is_train=False)
    
    # Evaluate test set
    print(f"Evaluating test set at: {args.test_dir}")
    predictions, labels, confidences, file_paths = evaluate_test_set(
        model, args.test_dir, transform, class_to_idx, device
    )
    
    if not predictions:
        print("No valid images found in test set!")
        return
    
    # Calculate metrics
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='weighted', zero_division=0)
    rec = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(labels, predictions)
    
    # Print results
    print("\nTest Set Evaluation Results:")
    print(f"Number of test images: {len(predictions)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    plot_confusion_matrix(cm, class_names)
    print("Confusion matrix saved to test_confusion_matrix.png")
    
    # Find misclassified examples
    misclassified = []
    for i in range(len(predictions)):
        if predictions[i] != labels[i]:
            misclassified.append((file_paths[i], labels[i], predictions[i], confidences[i]))
    
    # Sort by confidence (highest confidence errors first)
    misclassified.sort(key=lambda x: x[3], reverse=True)
    
    # Plot misclassified examples
    if misclassified:
        plot_misclassified_examples(misclassified, idx_to_class)
        print(f"Misclassified examples saved to misclassified_examples.png")
        print(f"Total misclassified: {len(misclassified)} of {len(predictions)} ({len(misclassified)/len(predictions):.1%})")
    else:
        print("No misclassified examples found!")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'image_path': file_paths,
        'true_class': [idx_to_class[label] for label in labels],
        'predicted_class': [idx_to_class[pred] for pred in predictions],
        'confidence': confidences,
        'correct': [pred == label for pred, label in zip(predictions, labels)]
    })
    
    results_df.to_csv(args.output, index=False)
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()
