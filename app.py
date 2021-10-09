#Import Librarires
from flask import Flask, render_template, request
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Import resnet model for feature extraction
extract_model = ResNet50(include_top=True)
last = extract_model.layers[-2].output
image_features_extract_model = Model(inputs = extract_model.input,outputs = last)

#Import the CNN RNN model  
reconstructed_model = load_model("model_ImageCap_Umair.h5")

#Import tokenizor from pickle file
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

    global reconstructed_model, image_features_extract_model, tokenizer, image

    file = request.files['file1']

    file.save('static/file.jpg')

    print("="*50)
    print("IMAGE SAVED")

    #State vocab size, max length(max tokens)
    vocab_size = len(tokenizer.word_index) 
    max_tokens = 73
    START_TOKEN = '<start> '
    END_TOKEN = ' <end>'
    token_start = tokenizer.word_index[START_TOKEN.strip()]
    token_end = tokenizer.word_index[END_TOKEN.strip()]


    #Import the file, incase of an application the user will upload the file here
    image_path = 'static/file.jpg'

    #Presprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image = tf.keras.applications.resnet.preprocess_input(x)

    #Use the prediction coding,extarct image features from ResNet50 model and RNN features from ------>
    #Reconstructed model, use that model to predict
    encoder_input = image_features_extract_model.predict(image)
    encoder_input = tf.reshape(encoder_input,
                                (2048, ))

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

    #Print prediction    
    final = tokenizer.sequences_to_texts([output])
    print(final)
    
    return render_template('after.html', final = final)

if __name__ == "__main__":
    app.run(debug=True)