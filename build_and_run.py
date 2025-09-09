import os
import pickle
import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.utils import to_categorical, plot_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(DATASET_DIR, 'Images')
CAPTIONS_FILE = os.path.join(DATASET_DIR, 'captions.txt')

FEATURES_FILE = os.path.join(MODELS_DIR, 'features.pkl')
TOKENIZER_FILE = os.path.join(MODELS_DIR, 'tokenizer.pkl')
MODEL_FILE = os.path.join(MODELS_DIR, 'caption_model.h5')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
def extract_features(directory):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    print("Extracting features from images...")
    for img_name in tqdm(image_files):
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = os.path.splitext(img_name)[0]
        features[image_id] = feature
    return features

if not os.path.exists(FEATURES_FILE):
    image_features = extract_features(IMAGES_DIR)
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump(image_features, f)
    print(f"Features extracted and saved to {FEATURES_FILE}")
else:
    print(f"Loading existing features from {FEATURES_FILE}...")
    with open(FEATURES_FILE, 'rb') as f:
        image_features = pickle.load(f)

def load_and_clean_captions(filename, features):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    captions = {}
    for line in lines:
        parts = line.split(',')
        if len(parts) < 2:
            continue
        image_id, caption = parts[0], ','.join(parts[1:])
        image_id = os.path.splitext(image_id)[0]
        
        if image_id in features:
            if image_id not in captions:
                captions[image_id] = []
            
            desc = caption.strip().lower()
            desc = desc.translate(str.maketrans('', '', string.punctuation))
            words = [word for word in desc.split() if len(word) > 1 and word.isalpha()]
            cleaned_caption = ' '.join(words)
            final_caption = 'startseq ' + cleaned_caption + ' endseq'
            captions[image_id].append(final_caption)
            
    return captions

print("Loading and cleaning captions...")
all_captions = load_and_clean_captions(CAPTIONS_FILE, image_features)
all_desc = [d for key in all_captions for d in all_captions[key]]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_desc)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(d.split()) for d in all_desc)

with open(TOKENIZER_FILE, 'wb') as f:
    pickle.dump(tokenizer, f)

def data_generator(captions, features, tokenizer, max_length, vocab_size, batch_size=32):
    image_ids = list(captions.keys())
    while True:
        np.random.shuffle(image_ids)
        for i in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[i:i+batch_size]
            X1, X2, y = [], [], []
            for image_id in batch_ids:
                pic_features = features[image_id][0]
                for cap in captions[image_id]:
                    seq = tokenizer.texts_to_sequences([cap])[0]
                    for j in range(1, len(seq)):
                        in_seq, out_seq = seq[:j], seq[j]
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                        X1.append(pic_features)
                        X2.append(in_seq)
                        y.append(out_seq)
            yield ((np.array(X1), np.array(X2)), np.array(y))

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
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    print(model.summary())
    return model

print("Defining the model...")
model = define_model(vocab_size, max_length)

print("Starting training...")
epochs = 20
batch_size = 64
steps_per_epoch = len(all_captions) // batch_size

output_signature = (
    (tf.TensorSpec(shape=(None, 512), dtype=tf.float32),  
     tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
)

dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(all_captions, image_features, tokenizer, max_length, vocab_size, batch_size),
    output_signature=output_signature
)

model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)

model.save(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

def predict_caption(model, image_features, tokenizer, max_length):
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

print("\n--- Generating Example Caption ---")
example_image_id = list(all_captions.keys())[42]
image_path = os.path.join(IMAGES_DIR, example_image_id + '.jpg')

pic_features = image_features[example_image_id]
generated_caption = predict_caption(model, pic_features, tokenizer, max_length)

print(f"Generated Caption: {generated_caption}")
print("Original Captions:")
for cap in all_captions[example_image_id]:
    print(cap)
    
img = Image.open(image_path)
plt.imshow(img)
plt.title(f"Generated: {generated_caption}")
plt.axis('off')
plt.show()