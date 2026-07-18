#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for image captioning.
This module implements metrics for evaluating caption quality.
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Make sure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def calculate_bleu(references, hypotheses, max_n=4):
    """
    Calculate BLEU score for a set of references and hypotheses.
    
    Args:
        references (list): List of reference lists (multiple references per sample)
        hypotheses (list): List of hypothesis lists (one per sample)
        max_n (int): Maximum n-gram to consider
        
    Returns:
        list: BLEU scores for different n-grams (BLEU-1, BLEU-2, etc.)
    """
    def _tokenize(text):
        return text.split() if isinstance(text, str) else text

    tokenized_references = [
        [_tokenize(ref) for ref in refs] for refs in references
    ]
    tokenized_hypotheses = [_tokenize(hyp) for hyp in hypotheses]

    smoothing = SmoothingFunction().method1

    bleu_scores = []
    for n in range(1, max_n + 1):
        weights = tuple(1.0 / n for _ in range(n))
        score = corpus_bleu(
            tokenized_references,
            tokenized_hypotheses,
            weights=weights,
            smoothing_function=smoothing
        )
        bleu_scores.append(score)

    return bleu_scores

def calculate_metrics(model, dataloader, vocab, device='cuda', max_samples=None, beam_size=1):
    """
    Calculate evaluation metrics for the captioning model.
    
    Args:
        model (nn.Module): Image captioning model
        dataloader (DataLoader): Data loader (should return image, caption, image_id)
        vocab (Vocabulary): Vocabulary object
        device (str): Device to use ('cuda' or 'cpu')
        max_samples (int, optional): Maximum number of samples to evaluate
        beam_size (int): Beam size for caption generation
        
    Returns:
        float: BLEU-4 score
    """
    model.eval()
    
    # Initialize reference and hypothesis lists
    references_by_id = defaultdict(list)
    hypotheses_by_id = {}
    
    # Generate captions
    with torch.no_grad():
        for i, (image, caption, image_id) in enumerate(tqdm(dataloader, desc="Generating captions")):
            # Process only max_samples if specified
            if max_samples is not None and i >= max_samples:
                break
            
            # Move image to device
            image = image.to(device)
            
            # Generate caption
            predicted_ids = model.generate_caption(
                image,
                beam_size=beam_size
            )
            
            # Convert to tokens
            predicted_caption = vocab.decode(predicted_ids, join=True, remove_special=True)
            
            # Get reference caption
            reference_caption = vocab.decode(caption[0], join=True, remove_special=True)
            
            # Store results
            image_id = image_id[0]  # Get string from list
            references_by_id[image_id].append(reference_caption)
            hypotheses_by_id[image_id] = predicted_caption
    
    # Prepare data for BLEU calculation
    references = [references_by_id[image_id] for image_id in hypotheses_by_id.keys()]
    hypotheses = [hypotheses_by_id[image_id] for image_id in hypotheses_by_id.keys()]
    
    # Calculate BLEU score
    bleu_scores = calculate_bleu(references, hypotheses)
    
    # Return BLEU-4 score (or highest available)
    return bleu_scores[-1]