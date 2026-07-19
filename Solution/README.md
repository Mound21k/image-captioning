# Image Captioning — Reference Solution

This folder contains a **complete, working reference implementation** of the CNN-RNN image captioning assignment defined in the [repository root](../README.md). All `TODO` sections from the base template have been filled in — the code here trains and runs end-to-end.

## Overview

The model follows a classic encoder–decoder architecture for image captioning:

1. **Encoder (CNN)** — extracts a fixed-size feature vector from an input image using a pretrained backbone (ResNet18, ResNet50, MobileNetV2, or Inception V3).
2. **Decoder (RNN)** — an LSTM or GRU that consumes the image feature as its initial state and generates a caption one word at a time.

Together these form the `CaptionModel`, trained with teacher forcing and evaluated with BLEU.

## Structure

```
Solution/
├── data/
│   └── download_flickr.py     # Downloads and preprocesses the Flickr8k dataset
├── models/
│   ├── encoder.py              # EncoderCNN: pretrained CNN backbone + projection layer
│   ├── decoder.py               # DecoderRNN: embedding + LSTM/GRU + output projection
│   ├── caption_model.py         # CaptionModel: combines encoder + decoder, greedy & beam search
│   └── config.json              # Default training/model hyperparameters
├── utils/
│   ├── dataset.py               # PyTorch Dataset / DataLoader for image-caption pairs
│   ├── vocabulary.py            # Vocabulary building and tokenization
│   ├── trainer.py               # Training and validation loop
│   └── metrics.py               # BLEU and other evaluation metrics
├── notebooks/
│   ├── 1_Data_Exploration.ipynb        # Dataset stats, caption/vocab analysis
│   ├── 2_Feature_Extraction.ipynb      # CNN feature extraction comparison
│   ├── 3_Model_Training.ipynb          # Full model training
│   └── 4_Evaluation_Visualization.ipynb # BLEU scoring, sample captions, greedy vs. beam search
└── README.md
```

## Setup

From the repository root:

```bash
git clone https://github.com/Mound21k/image-captioning.git
cd image-captioning
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download and prepare the Flickr8k dataset:

```bash
python Solution/data/download_flickr.py --data_dir ./data
```

## Running the Solution

Work through the notebooks in order, or run the underlying modules directly.

```bash
jupyter notebook Solution/notebooks/1_Data_Exploration.ipynb
jupyter notebook Solution/notebooks/2_Feature_Extraction.ipynb
jupyter notebook Solution/notebooks/3_Model_Training.ipynb
jupyter notebook Solution/notebooks/4_Evaluation_Visualization.ipynb
```

1. **Data Exploration** — inspects the dataset, builds and saves the vocabulary.
2. **Feature Extraction** — extracts and caches CNN features for each image using the configured backbone.
3. **Model Training** — trains the `CaptionModel` with teacher forcing, tracks train/val loss, and saves checkpoints.
4. **Evaluation & Visualization** — generates captions on held-out images, computes BLEU scores, and compares greedy vs. beam search decoding.

### Configuration

Default hyperparameters live in `models/config.json`:

| Key | Default | Description |
|---|---|---|
| `encoder_model` | `resnet18` | CNN backbone (`resnet18`, `resnet50`, `mobilenet_v2`, `inception_v3`) |
| `decoder_type` | `lstm` | RNN cell type (`lstm` or `gru`) |
| `embed_size` | `256` | Embedding / feature projection dimension |
| `hidden_size` | `512` | RNN hidden state size |
| `num_layers` | `1` | Number of RNN layers |
| `dropout` | `0.5` | Dropout probability |
| `batch_size` | `32` | Training batch size |
| `learning_rate` | `0.0003` | Optimizer learning rate |
| `num_epochs` | `15` | Max training epochs |
| `early_stopping_patience` | `5` | Epochs without val improvement before stopping |
| `vocab_size` | `2986` | Vocabulary size built from the dataset |

## Implementation Notes

- **Encoder** (`models/encoder.py`): loads a pretrained CNN, strips its final classification layer, and projects the resulting features through a `Linear → BatchNorm1d → ReLU → Dropout` block to `embed_size`. The backbone is frozen by default (`trainable=False`).
- **Decoder** (`models/decoder.py`): embeds caption tokens, feeds them through an LSTM/GRU initialized from the image feature, and projects hidden states to vocabulary logits.
- **CaptionModel** (`models/caption_model.py`): wires the encoder and decoder together, supporting teacher-forced training and both greedy and beam-search inference.

## Results

With the default configuration, the trained model can achieve approximately:
- BLEU-1: ~0.60–0.65
- BLEU-4: ~0.20–0.25

## Relationship to the Repository Root

The Python modules and notebooks in the repository root (`../models`, `../utils`, `../notebooks`, `../data`) are the **assignment template** with `TODO` placeholders for students to complete. This `Solution/` folder is the same structure with every `TODO` implemented — use it as a reference or answer key, not as the starting point for the assignment.
