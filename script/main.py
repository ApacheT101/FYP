import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import TweetTokenizer
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import json
import pickle
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from spellchecker import SpellChecker
import enchant 
from tensorflow.python.framework import ops
import tflearn
nltk.download('stopwords')

tokens = []

words = []
labels = []
docs_x = []
docs_y = []


training = []
output = []
#read the data
data = pd.read_csv(r"C:\Users\60122\Downloads\FYP\dataset.csv")

#filter the intent
data = data[['cancel_order', 'check_refund_policy', 'complaint', 'contact_customer_service', 'contact_human_agent', 'delete_account', 'delivery_options', 'delivery_period', 'edit_account', 'get_refund', 'recover_password','review']]


#create lists and store intents, BoW model for sentences and labels

def Modeltokenizer(tokenizer,data,corpus,vector,label,intent):
    #loop through dataset
     for column in data:
    
         for row in data[column]:
             #convert r to string and lowercase it 
             r = str(row)
             # if r is nan , skip it , otherwise add both data and intent into list 
             if r != 'nan' :
               tokens = tokenizer.tokenize(r)
             #add each tokens to list 
               corpus.extend(tokens)
               vector.append(tokens)
               label.append(column)
             else:
                continue
            
             if column not in intent:
	             intent.append(column)
 
    
def remove_stopwords_and_stemmization(stopwords,stemmer):
    #create a stopwords set from nltk
    words = [stemmer.stem(w.lower()) for w in words if w not in stopwords]
    words = sorted(list(set(words)))



    
def BoW_model(stemmer):
   

   out_empty = [0 for _ in range(len(labels))]
   for x,doc in enumerate(docs_x):
      bag = []
      wrds = [stemmer.stem(w) for w in doc]
      for w in words:
          if w in wrds:
              bag.append(1)
          else:
              bag.append(0)
      output_row = out_empty[:]
      output_row[labels.index(docs_y[x])] = 1
      training.append(bag)
      output.append(output_row)

  


def model():
    ops.reset_default_graph()
    net = tflearn.input_data(shape = [None, len(training[0])])
    net = tflearn.fully_connected(net,20)
    net = tflearn.fully_connected(net,20)
    net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch = 20, batch_size = 8, show_metric = True)
    model.save("model.tflearn")
    
 
    
 
training = np.array(training)
output = np.array(output)   
 
    
 
    
 

    