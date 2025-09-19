"""
Data Analysis and Preprocessing for Autism Detection Dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def analyze_datasets(self):
        """Analyze all available datasets"""
        print("🔍 Analyzing available datasets...")
        
        # Analyze ASD Data folder
        asd_data_path = self.data_dir / "ASD Data" / "ASD Data"
        if asd_data_path.exists():
            self._analyze_folder_structure(asd_data_path, "ASD Data")
        
        # Analyze Autistic Children dataset
        autistic_children_path = self.data_dir / "Autistic Children Facial Image Dataset"
        if autistic_children_path.exists():
            self._analyze_folder_structure(autistic_children_path, "Autistic Children")
        
        # Analyze Facial dataset
        facial_dataset_path = self.data_dir / "Facial dataset of autistic children"
        if facial_dataset_path.exists():
            self._analyze_folder_structure(facial_dataset_path, "Facial Dataset")
        
        # Analyze CSV files
        self._analyze_csv_files()
        
    def _analyze_folder_structure(self, folder_path, dataset_name):
        """Analyze folder structure and count images"""
        print(f"\n📁 {dataset_name}:")
        
        train_autistic = 0
        train_non_autistic = 0
        test_autistic = 0
        test_non_autistic = 0
        val_autistic = 0
        val_non_autistic = 0
        
        # Check for train folder
        train_paths = [
            folder_path / "Train",
            folder_path / "train"
        ]
        
        for train_path in train_paths:
            if train_path.exists():
                # Count autistic images
                for autistic_folder in ["autism", "autistic", "Autistic"]:
                    autistic_path = train_path / autistic_folder
                    if autistic_path.exists():
                        train_autistic = len(list(autistic_path.glob("*.jpg"))) + len(list(autistic_path.glob("*.png")))
                        break
                
                # Count non-autistic images
                for non_autistic_folder in ["tipical", "typical", "non_autistic", "Non_Autistic"]:
                    non_autistic_path = train_path / non_autistic_folder
                    if non_autistic_path.exists():
                        train_non_autistic = len(list(non_autistic_path.glob("*.jpg"))) + len(list(non_autistic_path.glob("*.png")))
                        break
                break
        
        # Check for test folder
        test_paths = [
            folder_path / "Test",
            folder_path / "test"
        ]
        
        for test_path in test_paths:
            if test_path.exists():
                # Count autistic images
                for autistic_folder in ["autism", "autistic", "Autistic"]:
                    autistic_path = test_path / autistic_folder
                    if autistic_path.exists():
                        test_autistic = len(list(autistic_path.glob("*.jpg"))) + len(list(autistic_path.glob("*.png")))
                        break
                
                # Count non-autistic images
                for non_autistic_folder in ["tipical", "typical", "non_autistic", "Non_Autistic"]:
                    non_autistic_path = test_path / non_autistic_folder
                    if non_autistic_path.exists():
                        test_non_autistic = len(list(non_autistic_path.glob("*.jpg"))) + len(list(non_autistic_path.glob("*.png")))
                        break
                break
        
        # Check for validation folder
        val_paths = [
            folder_path / "valid",
            folder_path / "val"
        ]
        
        for val_path in val_paths:
            if val_path.exists():
                # Count autistic images
                for autistic_folder in ["autistic", "Autistic"]:
                    autistic_path = val_path / autistic_folder
                    if autistic_path.exists():
                        val_autistic = len(list(autistic_path.glob("*.jpg"))) + len(list(autistic_path.glob("*.png")))
                        break
                
                # Count non-autistic images
                for non_autistic_folder in ["typical", "non_autistic", "Non_Autistic"]:
                    non_autistic_path = val_path / non_autistic_folder
                    if non_autistic_path.exists():
                        val_non_autistic = len(list(non_autistic_path.glob("*.jpg"))) + len(list(non_autistic_path.glob("*.png")))
                        break
                break
        
        total_autistic = train_autistic + test_autistic + val_autistic
        total_non_autistic = train_non_autistic + test_non_autistic + val_non_autistic
        
        print(f"  Train - Autistic: {train_autistic}, Non-autistic: {train_non_autistic}")
        print(f"  Test - Autistic: {test_autistic}, Non-autistic: {test_non_autistic}")
        print(f"  Validation - Autistic: {val_autistic}, Non-autistic: {val_non_autistic}")
        print(f"  Total - Autistic: {total_autistic}, Non-autistic: {total_non_autistic}")
        print(f"  Total Images: {total_autistic + total_non_autistic}")
        
        self.datasets[dataset_name] = {
            'train_autistic': train_autistic,
            'train_non_autistic': train_non_autistic,
            'test_autistic': test_autistic,
            'test_non_autistic': test_non_autistic,
            'val_autistic': val_autistic,
            'val_non_autistic': val_non_autistic,
            'total_autistic': total_autistic,
            'total_non_autistic': total_non_autistic
        }
    
    def _analyze_csv_files(self):
        """Analyze CSV files"""
        print(f"\n📊 CSV Files Analysis:")
        
        csv_files = [
            ("ASD Facial Image Dataset Combined/train.csv", "ASD Combined Train"),
            ("ASD Facial Image Dataset Combined/test.csv", "ASD Combined Test"),
            ("ASD Facial Image Dataset Combined/valid.csv", "ASD Combined Valid"),
            ("Facial dataset of autistic children/train.csv", "Facial Dataset Train"),
            ("Facial dataset of autistic children/test.csv", "Facial Dataset Test"),
            ("Facial dataset of autistic children/val.csv", "Facial Dataset Val"),
            ("Cleaned Dataset/cleaned_train.csv", "Cleaned Train"),
            ("Cleaned Dataset/cleaned_test.csv", "Cleaned Test"),
            ("Cleaned Dataset/cleaned_valid.csv", "Cleaned Valid")
        ]
        
        for csv_path, name in csv_files:
            full_path = self.data_dir / csv_path
            if full_path.exists():
                try:
                    df = pd.read_csv(full_path)
                    print(f"  {name}: {len(df)} rows")
                    if 'labels' in df.columns:
                        print(f"    Labels: {df['labels'].value_counts().to_dict()}")
                    elif 'label' in df.columns:
                        print(f"    Labels: {df['label'].value_counts().to_dict()}")
                except Exception as e:
                    print(f"  {name}: Error reading file - {e}")
    
    def sample_image_analysis(self, num_samples=5):
        """Analyze sample images from each dataset"""
        print(f"\n🖼️ Sample Image Analysis:")
        
        # Find sample images
        sample_paths = []
        
        # ASD Data samples
        asd_data_path = self.data_dir / "ASD Data" / "ASD Data" / "Train"
        if asd_data_path.exists():
            for folder in ["autism", "tipical"]:
                folder_path = asd_data_path / folder
                if folder_path.exists():
                    images = list(folder_path.glob("*.jpg"))[:2]
                    sample_paths.extend([(str(img), folder) for img in images])
        
        # Analyze sample images
        for img_path, label in sample_paths[:num_samples]:
            try:
                img = Image.open(img_path)
                print(f"  {os.path.basename(img_path)} ({label}): {img.size}, Mode: {img.mode}")
            except Exception as e:
                print(f"  Error analyzing {img_path}: {e}")
    
    def create_consolidated_dataset(self):
        """Create a consolidated dataset from the best available data"""
        print(f"\n🔄 Creating consolidated dataset...")
        
        # Use the largest dataset (Facial dataset of autistic children)
        source_dir = self.data_dir / "Facial dataset of autistic children"
        target_dir = self.data_dir / "data" / "processed"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create consolidated structure
        for split in ['train', 'test', 'val']:
            split_dir = target_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for label in ['autistic', 'non_autistic']:
                label_dir = split_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for split in ['train', 'test', 'val']:
            source_split = source_dir / split if split != 'val' else source_dir / 'val'
            target_split = target_dir / split
            
            if source_split.exists():
                # Copy Autistic images
                autistic_source = source_split / 'Autistic'
                autistic_target = target_split / 'autistic'
                if autistic_source.exists():
                    self._copy_images(autistic_source, autistic_target)
                
                # Copy Non_Autistic images
                non_autistic_source = source_split / 'Non_Autistic'
                non_autistic_target = target_split / 'non_autistic'
                if non_autistic_source.exists():
                    self._copy_images(non_autistic_source, non_autistic_target)
        
        print(f"✅ Consolidated dataset created in {target_dir}")
        return target_dir
    
    def _copy_images(self, source_dir, target_dir):
        """Copy images from source to target directory"""
        import shutil
        for img_file in source_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                target_file = target_dir / img_file.name
                shutil.copy2(img_file, target_file)
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n📋 Dataset Summary Report:")
        print("=" * 50)
        
        total_images = 0
        for dataset_name, stats in self.datasets.items():
            dataset_total = stats['total_autistic'] + stats['total_non_autistic']
            total_images += dataset_total
            print(f"\n{dataset_name}:")
            print(f"  Total Images: {dataset_total}")
            print(f"  Autistic: {stats['total_autistic']} ({stats['total_autistic']/dataset_total*100:.1f}%)")
            print(f"  Non-autistic: {stats['total_non_autistic']} ({stats['total_non_autistic']/dataset_total*100:.1f}%)")
        
        print(f"\nOverall Total Images: {total_images}")
        print("=" * 50)

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = DataAnalyzer("/Users/OFFISONG_1/Desktop/COMPANIES/AIRLAB_IT/autism-detection")
    
    # Run analysis
    analyzer.analyze_datasets()
    analyzer.sample_image_analysis()
    analyzer.generate_summary_report()
    
    # Create consolidated dataset
    consolidated_path = analyzer.create_consolidated_dataset()
    print(f"\n🎯 Recommended dataset: {consolidated_path}")
