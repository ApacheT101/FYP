from flask import Flask, render_template, request, jsonify, url_for
import nltk
import datetime
from nltk.stem.snowball import SnowballStemmer
import numpy as np


import tflearn
import tensorflow as tf
import random
import json
import pickle
import webbrowser

stemmer = SnowballStemmer(language='english')

with open("D:\Study Material\FYP\snowdata.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)


# Function to process input
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
# load the model
model.load("D:\Study Material\FYP\snowmodel.tflearn")

def get_bot_response(message):
    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message, words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]
        if results[result_index] > 0.5:
            return tag
        else:
            tag = "unknown"
            return tag
    return "Missing Data!"

