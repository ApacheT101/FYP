from flask import Flask, render_template, request, jsonify
import nltk
import datetime
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
from PIL import Image

stemmer = SnowballStemmer(language='english')

with open("snowdata.pickle","rb") as f:
	words, labels, training, output = pickle.load(f)


#Function to process input
def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)



net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,20)
net = tflearn.fully_connected(net,20)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
#load the model
print(model.load("snowmodel.tflearn"))
message = "cancel an order"
results = model.predict([bag_of_words(message,words)])[0]
print(results)
result_index = np.argmax(results)
print(result_index)


def get_bot_response():
	message = "cancel an order"
	if message:
		message = message.lower()
		results = model.predict([bag_of_words(message,words)])[0]
		result_index = np.argmax(results)
		tag = labels[result_index]
		if results[result_index] > 0.5:
			if tag == "cancel_order":
				response = "Your table has been booked successfully. Remaining tables: "
			elif tag == "available_tables":
				response = "There are " + " tables available at the moment."
				
			elif tag == "menu":
				day = datetime.datetime.now()
				day = day.strftime("%A")
				if day == "Monday":
					response = "Chef recommends: Steamed Tofu with Schezwan Peppercorn, Eggplant with Hot Garlic Sauce, Chicken & Chives, Schezwan Style, Diced Chicken with Dry Red Chilli, Schezwan Pepper"

				elif day == "Tuesday":
					response = "Chef recommends: Asparagus Fresh Shitake & King Oyster Mushroom, Stir Fried Chilli Lotus Stem, Crispy Fried Chicken with Dry Red Pepper, Osmanthus Honey, Hunan Style Chicken"

				elif day == "Wednesday":
					response = "Chef recommends: Baby Pokchoi Fresh Shitake Shimeji Straw & Button Mushroom, Mock Meat in Hot Sweet Bean Sauce, Diced Chicken with Bell Peppers & Onions in Hot Garlic Sauce, Chicken in Chilli Black Bean & Soy Sauce"

				elif day == "Thursday":
					response = "Chef recommends: Eggplant & Tofu with Chilli Oyster Sauce, Corn, Asparagus Shitake & Snow Peas in Hot Bean Sauce, Diced Chicken Plum Honey Chilli Sauce, Clay Pot Chicken with Dried Bean Curd Sheet"

				elif day == "Friday":
					response = "Chef recommends: Kailan in Ginger Wine Sauce, Tofu with Fresh Shitake & Shimeji, Supreme Soy Sauce, Diced Chicken in Black Pepper Sauce, Sliced Chicken in Spicy Mala Sauce"

				elif day == "Saturday":
					response = "Chef recommends: Kung Pao Potato, Okra in Hot Bean Sauce, Chicken in Chilli Black Bean & Soy Sauce, Hunan Style Chicken"

				elif day == "Sunday":
					response = "Chef recommends: Stir Fried Bean Sprouts & Tofu with Chives, Vegetable Thou Sou, Diced Chicken Plum Honey Chilli Sauce, Diced Chicken in Black Pepper Sauce"
		
				
		else:
			response = "I didn't quite get that, please try again."
		return str(response)
	return "Missing Data!"

get_bot_response()