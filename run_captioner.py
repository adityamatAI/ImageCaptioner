# run_captioner.py (Corrected by rebuilding model and loading weights)

import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

MODEL_FILE = os.path.join('models', 'caption_model.h5')
TOKENIZER_FILE = os.path.join('models', 'tokenizer.pkl')
with open(TOKENIZER_FILE, 'rb') as f:
    tokenizer = pickle.load(f)
max_length = 34
vocab_size = len(tokenizer.word_index) + 1 
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(512,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

print("Building model architecture...")
model = define_model(vocab_size, max_length)

print(f"Loading weights from {MODEL_FILE}...")
model.load_weights(MODEL_FILE)

print("Loading VGG16 for feature extraction...")
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_single_image_features(img_path, model):
    """Extracts features from a single image using the VGG16 model."""
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def predict_caption(model, image_features, tokenizer, max_length):
    """Generates a caption for an image given the pre-extracted features."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = ' '.join(in_text.split()[1:-1])
    return final_caption
image_to_test = 'abc.jpg' 
if not os.path.exists(image_to_test):
    print(f"Error: The image file '{image_to_test}' was not found.")
    print("Please make sure the image exists in the same directory as the script.")
else:
    print(f"Generating caption for {image_to_test}...")
    features = extract_single_image_features(image_to_test, vgg_model)
    caption = predict_caption(model, features, tokenizer, max_length)
    print(f"\nSUCCESS! The generated caption is: {caption}")
    img = Image.open(image_to_test)
    plt.imshow(img)
    plt.title(f"Generated Caption: {caption}")
    plt.axis('off')
    plt.show()