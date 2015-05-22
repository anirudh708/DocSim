import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import json
from collections import Counter

from flask import Flask,request
from flask import render_template
app = Flask(__name__)

def document_to_wordlist( review, remove_stopwords=False ):
	review_text = BeautifulSoup(review).get_text()
	review_text = re.sub("[^a-zA-Z]"," ", review_text)
	words = review_text.lower().split()
	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]
	return(words)

    
def document_to_sentences( review, tokenizer, remove_stopwords=False ):
	raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
	sentences = []
	for raw_sentence in raw_sentences:
	    if len(raw_sentence) > 0:
	        sentences.append(document_to_wordlist( raw_sentence, \
	          remove_stopwords ))
	return sentences,raw_sentences

@app.route('/')
def home():
	''' render a beautiful templete that takes N number
		of documents and Algo option.
	'''
	return render_template('home.html') 


@app.route('/visual',methods=['POST'])
def visual():
	''' get data depending on algo create a similarity matrix
		feed it to a templete for visulization
	'''
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	document = request.form['document']
	raw_sentences = tokenizer.tokenize(document.decode('utf8').strip())
	tfidf = TfidfVectorizer().fit_transform(raw_sentences)
	matrix = (tfidf * tfidf.T).A
	force = {}
	force["nodes"] = []
	force["links"] = [] 
	for each in raw_sentences:
	    temp={}
	    temp["name"] = each
	    force["nodes"].append(temp)
	for ((i,_),(j,_)) in itertools.combinations(enumerate(raw_sentences), 2):
	    temp = {}
	    temp["source"] = i
	    temp["target"] = j
	    temp["value"] = matrix[i][j]
	    force["links"].append(temp)
	graph = json.dumps(force)
	wordlist = []
	for each in raw_sentences:
		wordlist+=document_to_wordlist(each)
	print Counter(wordlist)
	return render_template('visual.html',graph = graph, sentences=raw_sentences)


if __name__ == '__main__':
	app.debug = True

	app.run()