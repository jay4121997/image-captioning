import os
# Just to not show tensorflow dubbing informations which clutter the output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pickle import load
from pickle import dump
from os import path
import os.path
from os import listdir
import string
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from numpy import argmax
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename

TOKENIZER_PATH = './tokenizer.pkl'
PREDICT_IMAGE = './uploads/example1.jpg'
MODEL_1='./modals_new/model_1.h5'
MODEL_10='./modals_new/model_10.h5'



app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 10
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'


def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


def extract_features_file(filename):
        model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        return feature


def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = ' '
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    in_text = in_text.rsplit(' ', 1)[0]
    return in_text


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        filename = 'example1.jpg'
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        print(filename)
    # return redirect(
    #     'uploads/'+filename
    # )
    

# load the tokenizer
    tokenizer = load(open(TOKENIZER_PATH, 'rb'))
    max_length = 34
    photo = extract_features_file(PREDICT_IMAGE)
    model = load_model(MODEL_1)
    description = generate_desc(model, tokenizer, photo, max_length)
    print('-------------------------------------------------------------')
    print('Model : 1')
    c1 = description
    print(description)

    print('Model : 2')
    model = load_model(MODEL_10)
    description = generate_desc(model, tokenizer, photo, max_length)
    c2 = description
    print(description)
    print('-------------------------------------------------------------')

    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index copy 2.html', files="../uploads/"+files[0], caption1=c1, caption2=c2 )

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

app.run(threaded=False)
