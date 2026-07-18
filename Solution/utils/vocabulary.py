#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vocabulary processing for image captioning.
This module handles building and managing the vocabulary for caption text.
"""

import nltk
from collections import Counter
import pandas as pd
from tqdm import tqdm
import os
import pickle
import string
import re

class Vocabulary:
    """
    Vocabulary class for processing and tokenizing caption text.
    Handles word-to-index and index-to-word mappings.
    """
    
    def __init__(self, freq_threshold=5, max_size=None):
        """
        Initialize the vocabulary.
        
        Args:
            freq_threshold (int): Minimum frequency for a word to be included in the vocabulary
            max_size (int, optional): Maximum vocabulary size (excluding special tokens)
        """
        # Special token indices
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        # Initialize mappings
        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }
        self.idx2word = {
            0: self.pad_token,
            1: self.start_token,
            2: self.end_token,
            3: self.unk_token
        }
        
        # Set initial counter index
        self.idx = 4
        
        # Set frequency threshold and maximum size
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        
        # Try to ensure NLTK's tokenizer is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def __len__(self):
        """Return the size of the vocabulary"""
        return len(self.word2idx)
    
    def tokenize(self, text):
        """
        Tokenize a caption text into a list of tokens.
        
        Args:
            text (str): Caption text to tokenize
            
        Returns:
            list: List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        
        # Split into tokens
        tokens = nltk.word_tokenize(text)
        
        # Filter out tokens with length <= 1
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def build_vocabulary(self, caption_series):
        """
        Build the vocabulary from a series of captions.
        
        Args:
            caption_series (pandas.Series): Series of caption texts
            
        Returns:
            self: The vocabulary object
        """
        counter = Counter()
        for caption in tqdm(caption_series, desc="Building vocabulary"):
            counter.update(self.tokenize(caption))

        sorted_words = sorted(counter.items(), key=lambda item: item[1], reverse=True)

        for word, freq in sorted_words:
            if freq < self.freq_threshold:
                continue
            if self.max_size is not None and (self.idx - 4) >= self.max_size:
                break

            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

        return self
    
    def numericalize(self, text, add_special_tokens=True):
        """
        Convert a caption text into a list of indices.
        
        Args:
            text (str): Caption text to convert
            add_special_tokens (bool): Whether to add start and end tokens
            
        Returns:
            list: List of token indices
        """
        tokens = self.tokenize(text)
        unk_idx = self.word2idx[self.unk_token]
        indices = [self.word2idx.get(token, unk_idx) for token in tokens]

        if add_special_tokens:
            indices = [self.word2idx[self.start_token]] + indices + [self.word2idx[self.end_token]]

        return indices
        
    def decode(self, indices, join=True, remove_special=True):
        """
        Convert a list of indices back to a caption text.
        
        Args:
            indices (list or torch.Tensor): List of token indices
            join (bool): Whether to join tokens into a string
            remove_special (bool): Whether to remove special tokens
            
        Returns:
            str or list: Caption text or list of tokens
        """
        # Convert tensor to list if needed
        if not isinstance(indices, list):
            indices = indices.tolist()
        
        # Convert indices to words
        tokens = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        
        # Remove special tokens if requested
        if remove_special:
            tokens = [token for token in tokens 
                     if token not in [self.pad_token, self.start_token, self.end_token]]
        
        # Join tokens if requested
        if join:
            return " ".join(tokens)
        
        return tokens
    
    def save(self, path):
        """
        Save the vocabulary to a file.
        
        Args:
            path (str): Path to save the vocabulary
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a vocabulary from a file.
        
        Args:
            path (str): Path to the vocabulary file
            
        Returns:
            Vocabulary: Loaded vocabulary object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


def build_vocab_from_captions(captions_path, output_dir, freq_threshold=5, max_size=None):
    """
    Build and save a vocabulary from a captions file.
    
    Args:
        captions_path (str): Path to the captions CSV file
        output_dir (str): Directory to save the vocabulary
        freq_threshold (int): Minimum frequency for a word to be included
        max_size (int, optional): Maximum vocabulary size
        
    Returns:
        Vocabulary: The built vocabulary
    """
    # Load captions
    captions_df = pd.read_csv(captions_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize vocabulary
    vocab = Vocabulary(freq_threshold=freq_threshold, max_size=max_size)
    
    # Build vocabulary
    vocab.build_vocabulary(captions_df['caption'])
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocabulary.pkl')
    vocab.save(vocab_path)
    
    print(f"Vocabulary built with {len(vocab)} words")
    print(f"Saved to {vocab_path}")
    
    return vocab