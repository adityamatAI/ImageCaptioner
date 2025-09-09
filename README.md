# AI Image Caption Generator

## 1. Introduction

This project implements an image captioning model using a deep learning architecture. It combines a pre-trained Convolutional Neural Network (CNN) for image feature extraction with a Recurrent Neural Network (RNN) to generate descriptive, human-like captions for any given image.

The model learns to understand the content of an image—the objects, their attributes, and the actions taking place—and translates this understanding into a natural language description.

-   `build_and_run.py`: This is the main script for the entire pipeline. It handles data preparation, feature extraction, model training, and saving the final assets (model weights, tokenizer). You run this script first to train the model from scratch.
-   `run_captioner.py`: This is a lightweight script for inference. Once the model is trained, you can use this script to quickly generate a caption for a new image without retraining.

## 2. How It Works

The model's architecture is an **Encoder-Decoder** framework, a popular choice for sequence-to-sequence tasks like this.

### Encoder
The encoder is a pre-trained **VGG16** model (trained on the ImageNet dataset). Its role is to "see" the image and extract its most important visual features. We remove the final classification layer of VGG16 and use the output as a compact, fixed-size vector (a 512-element feature vector in this case) that represents the image's content.

### Decoder
The decoder is a **Long Short-Term Memory (LSTM)** network. Its job is to take the image feature vector from the encoder and generate a text sequence (the caption) word by word. It learns the relationships between words and how to structure a sentence.

The process is as follows:
1.  The image features and a special "startseq" token are fed into the LSTM.
2.  The LSTM predicts the most likely first word of the caption.
3.  This predicted word is then fed back into the LSTM along with the image features.
4.  The LSTM predicts the second word.
5.  This process repeats until the model predicts a special "endseq" token or reaches the maximum caption length.


## 3. Features

-   **End-to-End Pipeline**: Includes scripts for both training and inference.
-   **Transfer Learning**: Leverages the powerful VGG16 model to avoid needing to train an image recognition model from the ground up.
-   **Data Generator**: Uses a Python generator for memory-efficient training, allowing the model to be trained on large datasets without loading everything into RAM.
-   **Clean & Modular Code**: The training and inference logic are separated into two distinct files for clarity and ease of use.
-   **Saved Assets**: The script saves the trained model weights and the vocabulary tokenizer, so you can easily run inference later.

