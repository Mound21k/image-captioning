#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RNN Decoder module for image captioning.
This module implements the decoder part of the image captioning system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    """
    RNN Decoder for generating captions from image features.
    Uses LSTM/GRU with word embeddings to generate captions word by word.
    """
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, rnn_type='lstm', dropout=0.5):
        """
        Initialize the decoder.
        
        Args:
            embed_size (int): Dimensionality of the input embeddings (from the encoder)
            hidden_size (int): Dimensionality of the RNN hidden state
            vocab_size (int): Size of the vocabulary
            num_layers (int): Number of layers in the RNN
            rnn_type (str): Type of RNN cell ('lstm' or 'gru')
            dropout (float): Dropout probability
        """
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Choose 'lstm' or 'gru'.")

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Projects image features into the RNN's initial hidden state, so the decoder
        # is conditioned on the image the same way during training and generation
        # (both _greedy_sample and _beam_search initialize hidden state from features).
        self.init_h = nn.Linear(embed_size, hidden_size)

    def _get_initial_hidden(self, features):
        """
        Build the RNN's initial hidden state from image features.

        Args:
            features (torch.Tensor): Image features from the encoder [batch_size, embed_size]

        Returns:
            tuple or torch.Tensor: Initial hidden state, shaped for self.rnn
        """
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        if self.rnn_type == 'lstm':
            c0 = torch.zeros_like(h0)
            return (h0, c0)
        return h0

    def forward(self, features, captions, hidden=None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            features (torch.Tensor): Image features from the encoder [batch_size, embed_size]
            captions (torch.Tensor): Ground truth captions [batch_size, seq_length]
            hidden (tuple or torch.Tensor, optional): Initial hidden state for the RNN
            
        Returns:
            torch.Tensor: Raw output scores for each word in the vocabulary
                        Shape: [batch_size, seq_length - 1, vocab_size], where
                        outputs[:, t] predicts captions[:, t + 1] (teacher forcing)
            tuple or torch.Tensor: Final hidden state of the RNN
        """
        if hidden is None:
            hidden = self._get_initial_hidden(features)

        # Feed all but the last token; each output step predicts the *next* token
        embedded = self.embedding(captions[:, :-1])

        outputs, hidden = self.rnn(embedded, hidden)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)

        return outputs, hidden
    
    def sample(self, features, max_length=20, start_token=1, end_token=2, temperature=1.0, beam_size=1):
        """
        Sample captions using either greedy search or beam search.
        
        Args:
            features (torch.Tensor): Image features from the encoder [batch_size, embed_size]
            max_length (int): Maximum caption length
            start_token (int): Index of the start token
            end_token (int): Index of the end token
            temperature (float): Sampling temperature (higher = more diverse outputs)
            beam_size (int): Beam size for beam search (1 = greedy search)
            
        Returns:
            list: List of generated caption token sequences
        """
        batch_size = features.size(0)
        device = features.device
        
        # If beam size is 1, use greedy sampling
        if beam_size == 1:
            return self._greedy_sample(features, max_length, start_token, end_token, temperature)
        else:
            return self._beam_search(features, max_length, start_token, end_token, beam_size)
    
    def _greedy_sample(self, features, max_length, start_token, end_token, temperature):
        """
        Greedy sampling (beam size = 1)
        """
        batch_size = features.size(0)
        device = features.device

        hidden = self._get_initial_hidden(features)
        inputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        generated = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            embedded = self.embedding(inputs)  # [batch_size, 1, embed_size]
            output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hidden_size]

            output = self.fc(output.squeeze(1))  # [batch_size, vocab_size]
            output = output / temperature

            predicted = output.argmax(dim=1)  # [batch_size]

            for i in range(batch_size):
                if not finished[i]:
                    generated[i].append(predicted[i].item())

            finished = finished | (predicted == end_token)
            inputs = predicted.unsqueeze(1)

            if finished.all():
                break

        sampled_ids = [torch.LongTensor(seq).to(device) for seq in generated]

        return sampled_ids
    
    def _beam_search(self, features, max_length, start_token, end_token, beam_size):
        """
        Beam search sampling for better caption quality.
        
        Note: This implementation is for batch size = 1 for simplicity.
        """
        device = features.device
        
        # We only support batch size 1 for beam search for simplicity
        if features.size(0) != 1:
            raise ValueError("Beam search currently only supports batch size 1")
        
        # Initialize with start token
        k = beam_size
        sequences = [([start_token], 0.0, None)]  # (sequence, score, hidden)
        
        # For the first step, use image features as initial hidden state
        if self.rnn_type == 'lstm':
            # For LSTM, we need to initialize (h0, c0)
            h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            hidden_init = (h0, c0)
        else:
            # For GRU, we just need to initialize h0
            hidden_init = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Run beam search
        for _ in range(max_length):
            all_candidates = []
            
            # Expand each current candidate
            for seq, score, hidden in sequences:
                # If sequence ended, keep it
                if seq[-1] == end_token:
                    all_candidates.append((seq, score, hidden))
                    continue
                
                # Forward pass through the model
                inputs = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)
                embed = self.embedding(inputs)
                
                # Initialize hidden state with image features for the first step
                if len(seq) == 1 and hidden is None:
                    output, hidden_next = self.rnn(embed, hidden_init)
                else:
                    output, hidden_next = self.rnn(embed, hidden)
                
                # Project to vocabulary
                output = self.fc(output.squeeze(1))  # [1, vocab_size]
                
                # Convert to probabilities
                output = F.log_softmax(output, dim=1)
                
                # Get top k candidates
                topk_probs, topk_indices = output.topk(k)
                
                # Create new candidates
                for i in range(k):
                    next_token = topk_indices[0, i].item()
                    next_score = score + topk_probs[0, i].item()
                    next_seq = seq + [next_token]
                    all_candidates.append((next_seq, next_score, hidden_next))
            
            # Select k best candidates
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:k]
            
            # Check if all sequences have ended
            if all(seq[-1] == end_token for seq, _, _ in sequences):
                break
        
        # Return the highest scoring sequence
        best_seq = sequences[0][0]
        return [torch.LongTensor(best_seq).to(device)]