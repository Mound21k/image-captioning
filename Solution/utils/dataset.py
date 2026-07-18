#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset utilities for image captioning.
This module implements PyTorch dataset classes for loading and preprocessing images and captions.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.vocabulary import Vocabulary

class FlickrDataset(Dataset):
    """
    PyTorch dataset class for the Flickr8k dataset.
    Loads images and their corresponding captions.
    """
    
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_length=50):
        """
        Initialize the dataset.
        
        Args:
            images_dir (str): Directory containing the images
            captions_file (str): Path to the captions CSV file
            vocab (Vocabulary): Vocabulary object for text processing
            transform (torchvision.transforms, optional): Image transformations
            max_length (int): Maximum caption length
        """
        self.images_dir = images_dir
        self.df = pd.read_csv(captions_file)
        self.vocab = vocab
        self.max_length = max_length
        
        # Define default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),  # Resize to 256x256
                transforms.CenterCrop(224),  # Center crop to 224x224
                transforms.ToTensor(),  # Convert to tensor (0-1)
                transforms.Normalize(  # Normalize with ImageNet mean and std
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, caption)
                image (torch.Tensor): Preprocessed image tensor
                caption (torch.Tensor): Caption token indices
        """
        row = self.df.iloc[idx]
        img_filename = row['image']
        caption_text = row['caption']

        image = Image.open(os.path.join(self.images_dir, img_filename)).convert('RGB')
        image = self.transform(image)

        caption_indices = self.vocab.numericalize(caption_text, add_special_tokens=True)

        pad_idx = self.vocab.word2idx[self.vocab.pad_token]
        if len(caption_indices) > self.max_length:
            caption_indices = caption_indices[:self.max_length]
        else:
            caption_indices = caption_indices + [pad_idx] * (self.max_length - len(caption_indices))

        caption = torch.LongTensor(caption_indices)

        return image, caption


class FlickrDatasetWithID(FlickrDataset):
    """
    Extended Flickr dataset that also returns image IDs.
    Useful for evaluation and visualization.
    """
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset with image ID.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, caption, image_id)
        """
        # Get base items
        image, caption = super().__getitem__(idx)
        
        # Get image ID
        img_name = self.df.iloc[idx]['image']
        
        return image, caption, img_name


def get_data_loaders(data_dir, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker threads for loading data
        pin_memory (bool): Whether to pin memory (useful for GPU training)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab)
    """
    # Define paths
    images_dir = os.path.join(data_dir, "processed", "images")
    train_captions = os.path.join(data_dir, "processed", "train_captions.csv")
    val_captions = os.path.join(data_dir, "processed", "val_captions.csv")
    test_captions = os.path.join(data_dir, "processed", "test_captions.csv")
    vocab_path = os.path.join(data_dir, "processed", "vocabulary.pkl")
    
    # Load vocabulary
    vocab = Vocabulary.load(vocab_path)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    # val_dataset uses FlickrDatasetWithID (not plain FlickrDataset) because
    # CaptionTrainer.calculate_metrics()/utils.metrics.calculate_metrics() and
    # notebook 3's sample-caption cell both iterate val_loader as (images, captions, image_ids).
    train_dataset = FlickrDataset(images_dir, train_captions, vocab, transform=train_transform)
    val_dataset = FlickrDatasetWithID(images_dir, val_captions, vocab, transform=val_transform)
    test_dataset = FlickrDatasetWithID(images_dir, test_captions, vocab, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Evaluate one image at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, vocab