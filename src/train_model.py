"""
Training Script for Autism Detection Model
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_analysis import DataAnalyzer
from models.autism_model import LightweightAutismCNN, AutismModelTrainer, create_data_loaders

def main():
    """Main training function"""
    print("🎯 Autism Detection Model Training")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Data paths
    base_dir = Path("/Users/OFFISONG_1/Desktop/COMPANIES/AIRLAB_IT/autism-detection")
    data_dir = base_dir / "data" / "processed"
    model_artifacts_dir = base_dir / "model_artifacts"
    
    # Create directories
    model_artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Analysis and Preprocessing
    print("\n📊 Step 1: Data Analysis and Preprocessing")
    print("-" * 40)
    
    analyzer = DataAnalyzer(base_dir)
    analyzer.analyze_datasets()
    
    # Create consolidated dataset if it doesn't exist
    if not data_dir.exists():
        print("\n🔄 Creating consolidated dataset...")
        data_dir = analyzer.create_consolidated_dataset()
    else:
        print(f"✅ Using existing consolidated dataset: {data_dir}")
    
    # Step 2: Create Data Loaders
    print("\n📦 Step 2: Creating Data Loaders")
    print("-" * 40)
    
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir, 
            batch_size=32, 
            num_workers=2  # Reduced for compatibility
        )
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return
    
    # Step 3: Initialize Model
    print("\n🧠 Step 3: Initializing Model")
    print("-" * 40)
    
    model = LightweightAutismCNN(num_classes=2)
    trainer = AutismModelTrainer(model, device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Step 4: Train Model
    print("\n🚀 Step 4: Training Model")
    print("-" * 40)
    
    try:
        best_val_acc = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=30,  # Reduced for faster training
            lr=0.001
        )
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Step 5: Evaluate Model
    print("\n📈 Step 5: Evaluating Model")
    print("-" * 40)
    
    try:
        predictions, targets = trainer.evaluate(test_loader)
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == np.array(targets))
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Generate classification report
        class_names = ['autistic', 'non_autistic']
        report = classification_report(
            targets, predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        print("\n📊 Classification Report:")
        print(classification_report(targets, predictions, target_names=class_names))
        
        # Save classification report
        report_path = model_artifacts_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("Autism Detection Model - Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(classification_report(targets, predictions, target_names=class_names))
            f.write("\n\nConfusion Matrix:\n")
            cm = confusion_matrix(targets, predictions)
            f.write(str(cm))
        
        print(f"💾 Classification report saved to {report_path}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(targets, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        cm_path = model_artifacts_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {cm_path}")
        plt.close()
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return
    
    # Step 6: Save Model and Artifacts
    print("\n💾 Step 6: Saving Model and Artifacts")
    print("-" * 40)
    
    try:
        # Save model
        model_path = model_artifacts_dir / "autism_detection_model.pth"
        trainer.save_model(model_path)
        
        # Plot training history
        history_path = model_artifacts_dir / "training_history.png"
        trainer.plot_training_history(history_path)
        
        # Save model architecture info
        model_info_path = model_artifacts_dir / "model_info.txt"
        with open(model_info_path, 'w') as f:
            f.write("Autism Detection Model Information\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model Architecture: LightweightAutismCNN\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB\n")
            f.write(f"Input Size: 224x224x3\n")
            f.write(f"Number of Classes: 2\n")
            f.write(f"Classes: ['autistic', 'non_autistic']\n")
            f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
            f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        
        print(f"📋 Model info saved to {model_info_path}")
        
        print(f"\n✅ All artifacts saved to {model_artifacts_dir}")
        print(f"   - Model: autism_detection_model.pth")
        print(f"   - Classification Report: classification_report.txt")
        print(f"   - Confusion Matrix: confusion_matrix.png")
        print(f"   - Training History: training_history.png")
        print(f"   - Model Info: model_info.txt")
        
    except Exception as e:
        print(f"❌ Error saving artifacts: {e}")
        return
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Model artifacts saved in: {model_artifacts_dir}")

if __name__ == "__main__":
    main()
