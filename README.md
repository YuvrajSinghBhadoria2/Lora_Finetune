# BERT Fine-tuning for Sentiment Analysis

This project demonstrates how to fine-tune a BERT model for binary sentiment classification using the IMDB movie reviews dataset. The model is trained to classify movie reviews as either positive or negative sentiment.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Saving the Model](#saving-the-model)

## Project Overview

This repository contains a Jupyter Notebook that walks through the process of fine-tuning a pre-trained BERT model for sentiment analysis. The model is trained on the IMDB dataset which contains 50,000 movie reviews labeled as positive or negative.

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Datasets library

## Installation

Install the required packages:

```bash
pip install transformers datasets torch
```

Or run the first cell in the Jupyter Notebook which will automatically install the dependencies:

```python
!pip install transformers datasets
```

## Usage

1. Clone this repository
2. Open the `Bert_Finetune.ipynb` notebook in Jupyter
3. Run all cells to train the model

Alternatively, you can run the notebook in Google Colab or any other Jupyter environment.

## Model Details

- **Base Model**: bert-base-uncased
- **Task**: Sequence Classification (Binary Sentiment Analysis)
- **Number of Labels**: 2 (Positive/Negative)

## Dataset

We use the IMDB movie reviews dataset which contains:
- 25,000 training samples
- 25,000 testing samples
- Binary labels (positive=1, negative=0)

The dataset is loaded using the Hugging Face Datasets library:

```python
dataset = load_dataset("imdb")
```

## Training Configuration

The model is trained with the following configuration:

```python
training_args = TrainingArguments(
    output_dir="./sentiment-bert",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

## Results

After training, the model achieves competitive performance on the IMDB test set. The evaluation results are printed at the end of training.

## Saving the Model

The trained model and tokenizer are saved to the `./sentiment-bert` directory:

```python
# Save model
trainer.save_model("./sentiment-bert")

# Save tokenizer
tokenizer.save_pretrained("./sentiment-bert")
```

You can later load the saved model using:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./sentiment-bert")
tokenizer = AutoTokenizer.from_pretrained("./sentiment-bert")
```