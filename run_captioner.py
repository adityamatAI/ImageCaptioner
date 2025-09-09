# --- run_captioner.py ---
# This script loads a pre-trained model and tokenizer to generate a caption for a single image.

# --- 1. Import Necessary Libraries ---
import os
import pickle
import numpy as np
from PIL import Image # Used for opening and handling image files.
import matplotlib.pyplot as plt # Used to display the image with its caption.
import tensorflow as tf # The backend framework for Keras.
from keras.models import Model # The Keras class used to define a model.
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add # All the layer types needed for the model architecture.
from keras.preprocessing.sequence import pad_sequences # A function to ensure all text sequences have the same length.
from keras.applications.vgg16 import VGG16, preprocess_input # VGG16 is the pre-trained image feature extractor.
from keras.preprocessing.image import load_img, img_to_array # Utilities for loading and converting images to arrays.

# --- 2. Load Pre-trained Model and Tokenizer ---
# Define paths to the saved model weights and the tokenizer file.
MODEL_FILE = os.path.join('models', 'caption_model.h5')
TOKENIZER_FILE = os.path.join('models', 'tokenizer.pkl')

# Load the tokenizer, which knows the vocabulary (word-to-index mapping) from the training data.
with open(TOKENIZER_FILE, 'rb') as f:
    tokenizer = pickle.load(f)

# Define model parameters. These must match the parameters used during training.
max_length = 34
vocab_size = len(tokenizer.word_index) + 1

# --- 3. Rebuild the Model Architecture ---
# This function defines the exact same model structure that was trained in the other script.
def define_model(vocab_size, max_length):
    # Image Feature Input: This branch processes the image features from VGG16.
    inputs1 = Input(shape=(512,)) # Expects a 512-element vector.
    fe1 = Dropout(0.5)(inputs1) # Dropout helps prevent overfitting.
    fe2 = Dense(256, activation='relu')(fe1) # A fully connected layer to process features.

    # Text Sequence Input: This branch processes the text caption generated so far.
    inputs2 = Input(shape=(max_length,)) # Expects a sequence of word indices.
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2) # Converts word indices into dense vectors (embeddings).
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2) # The LSTM layer processes the sequence of word embeddings.

    # Merge Branches: This is where image and text information are combined.
    decoder1 = add([fe2, se3]) # The 'add' layer fuses the visual and linguistic features.
    decoder2 = Dense(256, activation='relu')(decoder1)

    # Output Layer: Predicts the next word in the sequence.
    outputs = Dense(vocab_size, activation='softmax')(decoder2) # Softmax provides a probability for every word in the vocabulary.

    # Create the final model by specifying its inputs and outputs.
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# Create the model structure.
print("Building model architecture...")
model = define_model(vocab_size, max_length)

# Load the saved weights from the .h5 file into the newly created model structure.
print(f"Loading weights from {MODEL_FILE}...")
model.load_weights(MODEL_FILE)

# --- 4. Prepare the Image Feature Extractor (VGG16) ---
print("Loading VGG16 for feature extraction...")
# Load VGG16, pre-trained on ImageNet. `include_top=False` removes the final classification layer.
# `pooling='avg'` converts the final feature map into a single flat vector.
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# --- 5. Define Helper Functions for Inference ---
def extract_single_image_features(img_path, model):
    """Extracts features from a single image using the VGG16 model."""
    image = load_img(img_path, target_size=(224, 224)) # VGG16 requires 224x224 input.
    image = img_to_array(image) # Convert image to a NumPy array.
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # Add a batch dimension.
    image = preprocess_input(image) # Preprocess the image (e.g., normalize pixels) for VGG16.
    feature = model.predict(image, verbose=0) # Get the feature vector.
    return feature

def predict_caption(model, image_features, tokenizer, max_length):
    """Generates a caption for an image given the pre-extracted features."""
    in_text = 'startseq' # Start the generation with the 'startseq' token.
    # Loop to generate the caption word by word.
    for _ in range(max_length):
        # Convert the current text sequence into integers and pad it.
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Predict the next word. The inputs are the image features and the current text sequence.
        yhat = model.predict([image_features, sequence], verbose=0)
        # Get the index of the word with the highest probability. This is a "greedy search".
        yhat = np.argmax(yhat)
        # Convert the index back to a word.
        word = tokenizer.index_word.get(yhat, None)
        # Stop if we can't find the word or if the 'endseq' token is predicted.
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    # Clean up the final caption by removing the start and end tokens.
    final_caption = ' '.join(in_text.split()[1:-1])
    return final_caption

# --- 6. Run the Captioning Process ---
image_to_test = 'abc.jpg' # The name of the image you want to test.

# Check if the image file exists before proceeding.
if not os.path.exists(image_to_test):
    print(f"Error: The image file '{image_to_test}' was not found.")
    print("Please make sure the image exists in the same directory as the script.")
else:
    # If the image exists, generate the caption.
    print(f"Generating caption for {image_to_test}...")
    # Step 1: Extract features from the image.
    features = extract_single_image_features(image_to_test, vgg_model)
    # Step 2: Generate the caption using the model and the extracted features.
    caption = predict_caption(model, features, tokenizer, max_length)
    print(f"\nSUCCESS! The generated caption is: {caption}")
    # Step 3: Display the image and the generated caption.
    img = Image.open(image_to_test)
    plt.imshow(img)
    plt.title(f"Generated Caption: {caption}")
    plt.axis('off')
    plt.show()