#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training utilities for image captioning.
This module implements a trainer class for training the image captioning model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from utils.metrics import calculate_metrics

class CaptionTrainer:
    """
    Trainer class for image captioning models.
    Handles training, validation, checkpointing, and logging.
    """
    
    def __init__(self, model, train_loader, val_loader, vocab, 
                 device='cuda', learning_rate=3e-4, model_save_dir='models',
                 log_interval=100):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Image captioning model
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            vocab (Vocabulary): Vocabulary object
            device (str): Device to use ('cuda' or 'cpu')
            learning_rate (float): Learning rate for the optimizer
            model_save_dir (str): Directory to save model checkpoints
            log_interval (int): Logging interval (in batches)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.model_save_dir = model_save_dir
        self.log_interval = log_interval
        
        # Set device
        self.model = self.model.to(self.device)
        
        # Define loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word2idx[self.vocab.pad_token])
        
        # Define optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        
        # Define learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Separate batch-size-1 loader (with image ids) for BLEU computation.
        # self.val_loader is batched for efficient loss computation, but
        # calculate_metrics()/model.generate_caption() only support batch size 1.
        self.metrics_loader = DataLoader(
            self.val_loader.dataset,
            batch_size=1,
            shuffle=False
        )

        # Create model save directory
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_bleu': [],
            'lr': [],
            'epoch_times': []
        }
        
        # Initialize counters
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
        self.epochs_without_improvement = 0
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        batch_losses = []
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {self.current_epoch+1}")

        for i, (images, captions) in progress_bar:
            images = images.to(self.device)
            captions = captions.to(self.device)

            self.optimizer.zero_grad()

            outputs, _ = self.model(images, captions)
            targets = captions[:, 1:]

            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)

            progress_bar.set_postfix({'loss': f"{batch_loss:.4f}"})

            if (i + 1) % self.log_interval == 0:
                recent_avg = np.mean(batch_losses[-self.log_interval:])
                tqdm.write(f"Epoch {self.current_epoch+1}, Batch {i+1}/{len(self.train_loader)} - Loss: {recent_avg:.4f}")

        # Calculate average epoch loss
        avg_loss = epoch_loss / len(self.train_loader)
        tqdm.write(f"Epoch {self.current_epoch+1} - Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, generate_captions=False):
        """
        Validate the model on the validation set.
        
        Args:
            generate_captions (bool): Whether to generate and evaluate captions
            
        Returns:
            tuple: (val_loss, bleu_score)
        """
        self.model.eval()
        val_loss = 0.0

        # Metrics for caption generation (if enabled)
        bleu_score = 0.0

        with torch.no_grad():
            for images, captions, _ in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                captions = captions.to(self.device)

                outputs, _ = self.model(images, captions)
                targets = captions[:, 1:]

                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )
                val_loss += loss.item()

        if generate_captions:
            bleu_score = self.calculate_metrics()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(self.val_loader)
        tqdm.write(f"Epoch {self.current_epoch+1} - Val Loss: {avg_val_loss:.4f}" + 
                (f", BLEU-4: {bleu_score:.4f}" if generate_captions else ""))
        
        return avg_val_loss, bleu_score
    
    def calculate_metrics(self, num_samples=None):
        """
        Calculate BLEU score on validation or test set.
        
        Args:
            num_samples (int, optional): Number of samples to evaluate (None for all)
            
        Returns:
            float: BLEU-4 score
        """
        return calculate_metrics(
            self.model,
            self.metrics_loader,
            self.vocab,
            self.device,
            max_samples=num_samples
        )
    
    def train(self, epochs=10, early_stopping_patience=5, save_best_only=True, 
              evaluate_every=1, generate_every=5):
        """
        Train the model for multiple epochs.
        
        Args:
            epochs (int): Number of epochs to train
            early_stopping_patience (int): Patience for early stopping
            save_best_only (bool): Whether to save only the best model
            evaluate_every (int): Validate every N epochs
            generate_every (int): Generate captions every N epochs
            
        Returns:
            dict: Training history
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Training on device: {self.device}")
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
        self.epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Update current epoch
            self.current_epoch = epoch
            
            # Track epoch start time
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            if (epoch + 1) % evaluate_every == 0:
                generate_captions = (epoch + 1) % generate_every == 0
                val_loss, bleu_score = self.validate(generate_captions=generate_captions)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Check for improvement
                improved = False
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    improved = True
                    if save_best_only:
                        self.save_checkpoint(f"best_model_loss.pth")
                
                if generate_captions and bleu_score > self.best_bleu:
                    self.best_bleu = bleu_score
                    if save_best_only:
                        self.save_checkpoint(f"best_model_bleu.pth")
                
                # Early stopping
                if improved:
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs without improvement")
                    break
            else:
                # If not validating, set val_loss to NaN and bleu to 0
                val_loss = float('nan')
                bleu_score = 0.0
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_bleu'].append(bleu_score)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Calculate epoch duration
            epoch_time = time.time() - start_time
            self.history['epoch_times'].append(epoch_time)
            
            # Save checkpoint
            if not save_best_only:
                self.save_checkpoint(f"model_epoch_{epoch+1}.pth")
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}"
                  + (f", BLEU-4: {bleu_score:.4f}" if (epoch + 1) % generate_every == 0 else ""))
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best BLEU-4 score: {self.best_bleu:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename):
        """
        Save model checkpoint.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        checkpoint_path = os.path.join(self.model_save_dir, filename)
        
        torch.save({
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_bleu': self.best_bleu,
            'vocab': self.vocab,
            'history': self.history
        }, checkpoint_path)
        
        print(f"Model checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            
        Returns:
            self: The trainer object
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return self
        
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_bleu = checkpoint.get('best_bleu', 0.0)
        
        # Load vocabulary if available
        if 'vocab' in checkpoint:
            self.vocab = checkpoint['vocab']
        
        # Load history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")
        return self
    
    def plot_history(self):
        """
        Plot training history.
        
        Returns:
            tuple: (fig, axs) matplotlib figure and axes
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss
        axs[0, 0].plot(self.history['train_loss'], label='Training')
        axs[0, 0].plot(self.history['val_loss'], label='Validation')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot BLEU score
        epochs_with_bleu = [i for i, bleu in enumerate(self.history['val_bleu']) if bleu > 0]
        bleu_scores = [self.history['val_bleu'][i] for i in epochs_with_bleu]
        
        if bleu_scores:
            axs[0, 1].plot(epochs_with_bleu, bleu_scores)
            axs[0, 1].set_title('BLEU-4 Score')
            axs[0, 1].set_xlabel('Epoch')
            axs[0, 1].set_ylabel('BLEU-4')
            axs[0, 1].grid(True)
        else:
            axs[0, 1].text(0.5, 0.5, 'No BLEU scores recorded', 
                         ha='center', va='center', transform=axs[0, 1].transAxes)
        
        # Plot learning rate
        axs[1, 0].plot(self.history['lr'])
        axs[1, 0].set_title('Learning Rate')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Learning Rate')
        axs[1, 0].set_yscale('log')
        axs[1, 0].grid(True)
        
        # Plot epoch duration
        axs[1, 1].plot(self.history['epoch_times'])
        axs[1, 1].set_title('Epoch Duration')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Duration (s)')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        return fig, axs