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

        if self.model_name == 'resnet18':
            cnn = models.resnet18(pretrained=pretrained)
            self.feature_size = cnn.fc.in_features
            cnn.fc = nn.Identity()
        elif self.model_name == 'resnet50':
            cnn = models.resnet50(pretrained=pretrained)
            self.feature_size = cnn.fc.in_features
            cnn.fc = nn.Identity()
        elif self.model_name == 'mobilenet_v2':
            cnn = models.mobilenet_v2(pretrained=pretrained)
            self.feature_size = cnn.classifier[-1].in_features
            cnn.classifier = nn.Identity()
        elif self.model_name == 'inception_v3':
            cnn = models.inception_v3(pretrained=pretrained, aux_logits=False)
            self.feature_size = cnn.fc.in_features
            cnn.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        self.cnn = cnn

        # Projection layer: linear + batchnorm (normalization) + ReLU (activation) + dropout (regularization)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_size, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        if not trainable:
            for param in self.cnn.parameters():
                param.requires_grad = False

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