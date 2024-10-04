# BERT-based Text Classification using TensorFlow

## Project Overview

This project implements a text classification model using **BERT (Bidirectional Encoder Representations from Transformers)** pre-trained model to classify emotions. The model is built using **TensorFlow** and fine-tuned on the **SetFit/emotion** dataset, which consists of textual data labeled with different emotions.

The model is implemented to classify six different emotion labels: **joy, sadness, anger, fear, love, and surprise**.

## Tech Stack
- **Programming Language**: Python
- **Framework**: TensorFlow and Hugging Face Transformers
- **Key Libraries**:
  - `transformers`: Provides the BERT pre-trained model (`bert-base-uncased`) and tokenizer.
  - `datasets`: For loading the SetFit emotion dataset.
  - `tensorflow`: Used for building and training the neural network.
  - `tensorflow.keras.callbacks`: For implementing learning rate scheduler, early stopping, and checkpoint saving.

## Dataset

- **Dataset Name**: [SetFit/emotion](https://huggingface.co/datasets/SetFit/emotion)
- **Number of Classes**: 6 (joy, sadness, anger, fear, love, surprise)
- **Task**: Classifying input text into one of the emotion labels.
- **Training-Testing Split**: The dataset is split into training and testing sets during preprocessing.

## Model Architecture

The model is based on the **BERT architecture**, and its key components include:

1. **Pre-trained BERT**: Used as the core of the model to generate text embeddings.
2. **Fully Connected Layer**: A dense layer with 6 neurons (equal to the number of emotion classes) and a softmax activation function for classification.

The model uses the `bert-base-uncased` variant of BERT, which is a version of BERT that has been pre-trained on a large corpus of text and outputs lowercased text.

### Layers
- **BERT Embedding Layer**: Extracts feature representations from the input text.
- **Dense Layer**: A fully connected layer that produces the final classification logits for each class.

### Activation Function
- **Softmax**: Used in the output layer to return probability distributions for each class.

## Training

### Hyperparameters
- **Batch Size**: 64
- **Epochs**: 5
- **Optimizer**: Adam with a learning rate of `1e-5`
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

### Callbacks Used
- **Learning Rate Scheduler**: `ReduceLROnPlateau` reduces the learning rate when the validation loss plateaus.
- **Early Stopping**: Stops the training process if the validation loss does not improve for 3 consecutive epochs.
- **Model Checkpoint**: Saves the best model based on validation loss during training.

### References
- **BERT Pretrained Model**: BERT on Hugging Face
- **SetFit Emotion Dataset**: SetFit/emotion
- **TensorFlow Documentation**: TensorFlow
- **Hugging Face Transformers**: Transformers
