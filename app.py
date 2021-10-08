from flask import Flask, render_template, request
import tensorflow as tf

import matplotlib.pyplot as plt

import re
import numpy as np
import os
import time
import json
import gc
from glob import glob
from PIL import Image
import pickle
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GRU, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

extract_model = tf.keras.applications.ResNet50(include_top=True)
last = extract_model.layers[-2].output
image_features_extract_model = Model(inputs = extract_model.input,outputs = last)


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


embedding_dim = 512
embedding_size = 512
units = 512
vocab_size = len(tokenizer.word_index) 
max_length = 74
max_len = max_length -1
encoder_shape=2048
 
reconstructed_model = load_model("model_ImageCap_Umair.h5")



app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

    global reconstructed_model, image_features_extract_model, tokenizer, load_img, img_to_array, image

    file = request.files['file1']

    file.save('static/file.jpg')

    print("="*50)
    print("IMAGE SAVED")

    max_tokens = 73
    token_start = '<start >'
    token_end = ' <end>'
    def preprocess(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x /= 255.
        return x
    
    image = preprocess('static/file.jpg')

    encoder_input = image_features_extract_model.predict(image)

    encoder_input = np.expand_dims(encoder_input, axis=0)

    shape = (1, max_tokens)
    decoder_input = np.zeros(shape=shape, dtype=np.int)

    token_id = token_start
  
    output=[]

    count_tokens = 0

    while token_id != token_end and count_tokens < max_tokens:
        
        decoder_input[0, count_tokens] = token_id

        input_data ={'encoder_input':encoder_input ,'decoder_input': decoder_input}
        
        predict = reconstructed_model.predict(input_data)
        
        token_id = np.argmax(predict[0, count_tokens, :])
        
        output.append(token_id)
        
        count_tokens += 1
    
    final = tokenizer.sequences_to_texts([output])
    
    return render_template('after.html', final = final)

if __name__ == "__main__":
    app.run(debug=True)