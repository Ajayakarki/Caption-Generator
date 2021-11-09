from flask import Flask, render_template, request

import numpy as np
from numpy import array
from keras import Input, layers
from keras import optimizers
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.utils import to_categorical

import pickle as pkl
import tensorflow as tf



index_to_word = pkl.load(open("index_to_word.pickle", "rb"))
word_to_index = pkl.load(open("word_to_index.pickle", "rb"))
print("loaded")


vocab_size = len(index_to_word) + 1
vocab_size

max_length = 38


model = InceptionV3(weights='imagenet')
new_model = Model(model.input, model.layers[-2].output)





app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    global index_to_word, word_to_index, vocab_size, max_length, new_model


    #img = request.files['file1']

    #saved = img.save('static/file.jpg')

    file = request.files['file1']

    file.save('static/file.jpg')

    def preprocess(image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def encode(image):
        image = preprocess(image) 
        fea_vec = new_model.predict(image) 
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        return fea_vec

    encoding_train = encode('static/file.jpg')
    train_features = encoding_train

    final_model = tf.keras.models.load_model('image_caption_model.h5')


    def predict_caption(photo):
        in_text = 'stratsequence'
        for i in range(max_length):
            sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
            sequence = pad_sequences([sequence], maxlen=max_length)
        
            yhat = final_model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = index_to_word[yhat]
            in_text += ' ' + word
            if word == 'endsequence':
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final
    

    photo = train_features.reshape(1,2048)

    caption = predict_caption(photo)



    return render_template('predict.html', data=caption)











if __name__ == "__main__":
    app.run(debug=True)