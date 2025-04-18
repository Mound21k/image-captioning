#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN Encoder module for image captioning.
This module implements the encoder part of the image captioning system.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder for extracting feature representations from images.
    Uses a pre-trained CNN backbone with the classification head removed.
    """
    
    def __init__(self, model_name='resnet18', embed_size=256, pretrained=True, trainable=False):
        """
        Initialize the encoder.
        
        Args:
            model_name (str): Name of the CNN backbone to use
                Supported models: 'resnet18', 'resnet50', 'mobilenet_v2', 'inception_v3'
            embed_size (int): Dimensionality of the output embeddings
            pretrained (bool): Whether to use pre-trained weights
            trainable (bool): Whether to fine-tune the CNN backbone
        """
        super(EncoderCNN, self).__init__()
        
        self.model_name = model_name.lower()
        self.embed_size = embed_size
        
        # TODO: Initialize and configure the CNN backbone based on model_name
        # 1. Create the CNN model using torchvision.models with pretrained weights if specified
        # 2. Store the feature dimension size (before the final classifier)
        # 3. Remove the classifier/fully-connected layer and replace with nn.Identity()
        # Hint: Look at model architectures to find the classifier attribute name and output feature size
        
        # TODO: Create a projection layer to transform CNN features to embed_size
        # The projection should include normalization, activation, and regularization
        
        # TODO: Freeze CNN parameters if trainable is False
        # This prevents the CNN backbone from being updated during training
    
    def forward(self, images):
        """
        Forward pass to extract features from images.
        
        Args:
            images (torch.Tensor): Batch of input images [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: Image features [batch_size, embed_size]
        """
        # Extract features from CNN
        features = self.cnn(images)
        
        # Project features to the specified embedding size
        features = self.projection(features)
        
        return features
    
    def get_feature_size(self):
        """Returns the raw feature size of the CNN backbone"""
        return self.feature_size