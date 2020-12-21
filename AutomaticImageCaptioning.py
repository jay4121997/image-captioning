# -*- coding: utf-8 -*-
"""project_pr.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18jjrgFqWKlBQD1ZqFX3uCACXjbirA--G
"""

#Importig Libraries

import os
# Just to not show tensorflow dubbing informations which clutter the output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from numpy import argmax
from keras.layers.merge import add
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
import string
from os import listdir
import os.path
from os import path
from pickle import dump
from pickle import load


# Defining paths
DATASET_DIR = 'Flicker8k_Dataset'
IMAGEFEATURES = "ImageFeatures.pkl"
TOKEN = 'Flickr8k_text/Flickr8k.token.txt'
TRAINING_TEXT='Flickr8k_text/Flickr_8k.trainImages.txt'
MODEL_1 = "./models_new/model_1.h5"
MODEL_10 = './models_new/model_10.h5'
PREDIC_IMAGE= 'example1.jpg'


# Function to extract features from the images
def imageData(directory):
    model = VGG16()    
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output) # adjusting VGG model    
    print(model.summary()) 
    counter = 0
    features = dict()
    for fileName in listdir(directory):
        counter = counter+1
        filename = directory + '/' + fileName
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = fileName.split('.')[0]
        features[image_id] = feature
        print('>%s' % fileName)
        print(counter)
    return features


# Pre processing images
# This code only run once ImageFeatures.pkl exist images won't be preprocessed to save execution time
if not path.exists(IMAGEFEATURES):
    directory = DATASET_DIR
    features = imageData(directory)
    print('Extracted Features: %d' % len(features))
    # save to file
    dump(features, open(IMAGEFEATURES, 'wb'))
else:
    print('Using saved ImageFeatures.pkl')


# Get the text data from file (captions)
def get_text(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# mappting images to their descriptions for training
def loadingDescription(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping


def cleanupDesc(descriptions):
    # removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)

# Split word of image descriptions
def splitWords(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

# Save captions into files with images
def saveCaptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


filename = TOKEN
doc = get_text(filename)
descriptions = loadingDescription(doc)
print('Loaded: %d ' % len(descriptions))
cleanupDesc(descriptions)
vocabulary = splitWords(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
saveCaptions(descriptions, 'descriptions.txt')


# Get images 
def getImageNames(filename):
    doc = get_text(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# Getting cleaned descriptions without punctions
def getCleanText(filename, dataset):
    doc = get_text(filename)
    captions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in captions:
                captions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            captions[image_id].append(desc)
    return captions


# Getting features of photo
def getPhotoData(filename, dataset):
    allFeatures = load(open(filename, 'rb'))
    features = {k: allFeatures[k] for k in dataset}
    return features

# Converting description into lines
def dictToLines(captions):
    all_desc = list()
    for key in captions.keys():
        [all_desc.append(d) for d in captions[key]]
    return all_desc


# Creating tokens for given captions
def createToken(captions):
    lines = dictToLines(captions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# Get caption with max words
def max_length(captions):
    lines = dictToLines(captions)
    return max(len(d.split()) for d in lines)

# get images, input sequences and output words for given image

def getOutputWords(tokenizer, max_length, descriptionsList, image, vocab_size):
    X1, X2, y = list(), list(), list()
    for desc in descriptionsList:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(image)
            X2.append(in_seq)
            y.append(out_seq)
    print(array(X1),'/n', array(X2),'/n', array(y))
    return array(X1), array(X2), array(y)

# Captioning model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
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
    model.summary()   
    return model

# data generator, intended to be used in a call to model.fit_generator()


def dataForModel(descriptions, photos, tokenizer, max_length, vocab_size):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = getOutputWords(
                tokenizer, max_length, desc_list, photo, vocab_size)
            yield [[in_img, in_seq], out_word]


# Trainging models for 20 epochs 
# once a models are trained the training parts will be skipped to save execution time

if not path.exists(MODEL_1):
    filename = TRAINING_TEXT
    train = getImageNames(filename)
    print('Dataset: %d' % len(train))
    train_captions = getCleanText('descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_captions))
    train_features = getPhotoData('ImageFeatures.pkl', train)
    print('Photos: train=%d' % len(train_features))
    tokenizer = createToken(train_captions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    max_length = max_length(train_captions)
    print('Description Length: %d' % max_length)

    model = define_model(vocab_size, max_length)
    epochs = 20
    steps = len(train_captions)
    for i in range(epochs):
        generator = dataForModel(
            train_captions, train_features, tokenizer, max_length, vocab_size)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save('model_' + str(i) + '.h5')
        # IF YOU ARE NOT USING EXISTING MODEL PLEASE MOVE THE MODEL TO MODELS_NEW FOLDER AFTER THEY ARE SAVED HERE
else:
    print('Using existing model')

# Evelauate model


# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	in_text = 'Captions : '
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	for key, desc_list in descriptions.items():
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# This section is commented just to skip evaluation of model every time program runs

# prepare tokenizer on train set

# # load training dataset (6K)
# filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
# train = getImageNames(filename)
# print('Dataset: %d' % len(train))
# # descriptions
# train_descriptions = getCleanText('descriptions.txt', train)
# print('Descriptions: train=%d' % len(train_descriptions))
# # prepare tokenizer
# tokenizer = createToken(train_descriptions)
# vocab_size = len(tokenizer.word_index) + 1
# print('Vocabulary Size: %d' % vocab_size)
# # determine the maximum sequence length
# max_length = max_length(train_descriptions)
# print('Description Length: %d' % max_length)

# # prepare test set

# # load test set
# filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
# test = getImageNames(filename)
# print('Dataset: %d' % len(test))
# # descriptions
# test_descriptions = getCleanText('descriptions.txt', test)
# print('Descriptions: test=%d' % len(test_descriptions))
# # photo features
# test_features = getPhotoData('features.pkl', test)
# print('Photos: test=%d' % len(test_features))

# # load the model
# filename = MODEL_1
# model = load_model(filename)
# # evaluate model
# evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)


# create tokenizer
from keras.preprocessing.text import Tokenizer
from pickle import dump

# load training dataset (6K)
filename = TRAINING_TEXT
train = getImageNames(filename)
print('Dataset: %d' % len(train))
train_descriptions = getCleanText('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
tokenizer = createToken(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))




# create new model

# extract features from each photo in the directory
def extract_features_file(filename):
	model = VGG16()
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
photo = extract_features_file(PREDIC_IMAGE)
model = load_model(MODEL_1)
description = generate_desc(model, tokenizer, photo, max_length)
print('-------------------------------------------------------------')
print('Model : 1')
print(description)

print('Model : 2')
model = load_model(MODEL_10)
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
print('-------------------------------------------------------------')