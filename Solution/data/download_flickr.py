#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flickr8k dataset downloader and preprocessor.
This script downloads the Flickr8k dataset and organizes it for the image captioning task.
"""

import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
import shutil
import tarfile
import argparse
import re

def download_file(url, destination):
    """
    Downloads a file from a URL to a destination with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Path to save the downloaded file
    """
    if os.path.exists(destination):
        print(f"File already exists at {destination}. Skipping download.")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

def extract_zip(zip_path, extract_path):
    """
    Extracts a zip file to a destination folder.
    
    Args:
        zip_path (str): Path to the zip file
        extract_path (str): Path to extract the contents to
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f"Extracting {os.path.basename(zip_path)}"):
            zip_ref.extract(member, extract_path)

def extract_tar(tar_path, extract_path):
    """
    Extracts a tar file to a destination folder.
    
    Args:
        tar_path (str): Path to the tar file
        extract_path (str): Path to extract the contents to
    """
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        for member in tqdm(tar_ref.getmembers(), desc=f"Extracting {os.path.basename(tar_path)}"):
            tar_ref.extract(member, extract_path)

def process_captions(dataset_path):
    captions_path = os.path.join(
    dataset_path,
    "Flickr8k.token.txt"
    )

    if not os.path.exists(captions_path):
        captions_path = os.path.join(
            dataset_path,
            "Flickr8k_text",
            "Flickr8k.token.txt"
        )
    data = []

    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            image_and_id, caption = line.split('\t', 1)

            image_name = re.sub(
                r'(\.\d+)?#\d+$',
                '',
                image_and_id
            )

            data.append({
                'image': image_name,
                'caption': caption
            })

    df = pd.DataFrame(data)

    # Remove captions for images that do not exist
    images_dir = os.path.join(
        dataset_path,
        "Flicker8k_Dataset"
    )

    if os.path.exists(images_dir):
        available_images = set(os.listdir(images_dir))

        before = len(df)
        df = df[df["image"].isin(available_images)]
        after = len(df)

        print(
            f"Removed {before-after} captions "
            "with missing images"
        )

    output_dir = os.path.join(dataset_path, "processed")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "captions.csv")
    df.to_csv(output_path, index=False)

    print(f"Processed captions saved to {output_path}")

    return output_path

def organize_images(dataset_path):
    """
    Organizes the images into a clean directory structure.
    
    Args:
        dataset_path (str): Path to the dataset directory
    
    Returns:
        str: Path to the organized images directory
    """
    source_images_dir = os.path.join(dataset_path, "Flickr8k_Dataset", "Flicker8k_Dataset")
    
    if not os.path.exists(source_images_dir):
        source_images_dir = os.path.join(dataset_path, "Flicker8k_Dataset")  # Alternative path
    
    # Create clean output dir
    output_dir = os.path.join(dataset_path, "processed", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy images to new location
    for img_file in tqdm(os.listdir(source_images_dir), desc="Organizing images"):
        if img_file.startswith("._"):
            continue
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            source = os.path.join(source_images_dir, img_file)
            destination = os.path.join(output_dir, img_file)
            
            if not os.path.exists(destination):
                shutil.copy2(source, destination)
        
    
    print(f"Images organized in {output_dir}")
    return output_dir

def create_splits(dataset_path):
    """
    Creates train/val/test splits based on the official Flickr8k splits.
    
    Args:
        dataset_path (str): Path to the dataset directory
    """
    processed_dir = os.path.join(dataset_path, "processed")
    flickr_text_dir = os.path.join(dataset_path, "Flickr8k_text")
    
    if not os.path.exists(flickr_text_dir):
        flickr_text_dir = dataset_path
    
    def read_image_list(path):
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())

    train_images = read_image_list(os.path.join(flickr_text_dir, "Flickr_8k.trainImages.txt"))
    val_images = read_image_list(os.path.join(flickr_text_dir, "Flickr_8k.devImages.txt"))
    test_images = read_image_list(os.path.join(flickr_text_dir, "Flickr_8k.testImages.txt"))

    # Load caption data
    captions_df = pd.read_csv(os.path.join(processed_dir, "captions.csv"))

    train_df = captions_df[captions_df['image'].isin(train_images)]
    val_df = captions_df[captions_df['image'].isin(val_images)]
    test_df = captions_df[captions_df['image'].isin(test_images)]

    train_df.to_csv(os.path.join(processed_dir, "train_captions.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val_captions.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_captions.csv"), index=False)

    print(f"Created data splits: train ({len(train_df)} captions), val ({len(val_df)} captions), test ({len(test_df)} captions)")

def download_flickr8k(base_dir="./data"):
    """
    Downloads and prepares the Flickr8k dataset.
    
    Args:
        base_dir (str): Base directory to store the dataset
    
    Returns:
        dict: Dictionary with paths to the dataset components
    """
    dataset_path = os.path.join(base_dir, "flickr8k")
    os.makedirs(dataset_path, exist_ok=True)
    
    # URLs for Flickr8k dataset
    # Note: In a real implementation, you would use official download links
    # For this example, we're using placeholders that should be replaced with official sources
    images_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    images_zip = os.path.join(dataset_path, "Flickr8k_Dataset.zip")
    text_zip = os.path.join(dataset_path, "Flickr8k_text.zip")
    
    # Download dataset files
    download_file(images_url, images_zip)
    download_file(text_url, text_zip)
    
    # Extract dataset files
    extract_zip(images_zip, dataset_path)
    extract_zip(text_zip, dataset_path)
    
    # Process captions
    captions_path = process_captions(dataset_path)
    
    # Organize images
    images_path = organize_images(dataset_path)
    
    # Create data splits
    create_splits(dataset_path)
    
    return {
        "dataset_path": dataset_path,
        "images_path": images_path,
        "captions_path": captions_path,
        "processed_path": os.path.join(dataset_path, "processed")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare the Flickr8k dataset")
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Base directory to store the dataset")
    args = parser.parse_args()
    
    paths = download_flickr8k(args.data_dir)
    
    print("\nDataset preparation complete!")
    print(f"Dataset stored in: {paths['dataset_path']}")
    print(f"Processed images: {paths['images_path']}")
    print(f"Processed captions: {paths['captions_path']}")