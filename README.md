# LoRA Fine-tuning with DialoGPT

This project demonstrates how to use LoRA (Low-Rank Adaptation) to fine-tune a causal language model (DialoGPT) on the IMDB dataset. LoRA is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters while maintaining performance.

## Table of Contents
- [Project Overview](#project-overview)
- [What is LoRA?](#what-is-lora)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [LoRA Configuration](#lora-configuration)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Saving the Model](#saving-the-model)

## Project Overview

This project demonstrates how to use LoRA (Low-Rank Adaptation) to fine-tune a causal language model (DialoGPT) on the IMDB dataset. LoRA is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters while maintaining performance.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large language models. Instead of updating all model parameters, LoRA injects trainable low-rank matrices into the model's layers. This approach:

- Reduces trainable parameters by up to 99%
- Maintains model performance
- Requires less GPU memory
- Enables faster training times

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Datasets library
- PEFT (Parameter-Efficient Fine-Tuning) library
- TRL (Transformer Reinforcement Learning) library

## Installation

Install the required packages:

```bash
pip install transformers datasets peft trl torch
```

Or run the installation cells in the Jupyter Notebook:

```python
!pip install transformers datasets
!pip install peft
!pip install trl
!pip install torch
```

## Usage

1. Open the `Lora_Finetune.ipynb` notebook in Jupyter
2. Run all cells to train the model with LoRA adapters

Alternatively, you can run the notebook in Google Colab or any other Jupyter environment.

## Model Details

- **Base Model**: microsoft/DialoGPT-medium
- **Task**: Causal Language Modeling
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Data Type**: bfloat16
- **Device Map**: auto (automatically distributed across available devices)

## Dataset

We use the IMDB movie reviews dataset which contains:
- 25,000 training samples
- 25,000 testing samples

The dataset is loaded using the Hugging Face Datasets library:

```python
dataset = load_dataset("imdb")
```

## LoRA Configuration

The LoRA configuration uses the following parameters:

```python
lora_config = LoraConfig(
    r=16,                # Rank of the low-rank matrices
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.05,   # Dropout probability
    bias="none",         # Whether to train bias terms
    task_type="CAUSAL_LM",  # Task type
    target_modules=["c_attn", "c_proj"]  # Modules to apply LoRA to
)
```

## Training Configuration

The model is trained with the following configuration:

```python
training_args = TrainingArguments(
    output_dir="./lora-dialogpt",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
)
```

## Results

After training, the model achieves good performance on the IMDB dataset while only training a small subset of parameters. The evaluation results are printed at the end of training.

## Saving the Model

Only the LoRA adapters are saved to reduce storage requirements:

```python
model.save_pretrained("./lora-adapter")
```

To load the fine-tuned model later:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
```