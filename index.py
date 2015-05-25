import re
import os
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities

import itertools
import json
from collections import Counter

from flask import Flask,request
from flask import render_template
app = Flask(__name__)


def document_to_wordlist( review, remove_stopwords=False ):
	'''
		Takes a string and converts it to wordlist(list)
	'''
	review_text = BeautifulSoup(review).get_text()
	review_text = re.sub("[^a-zA-Z]"," ", review_text)
	words = review_text.lower().split()
	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]
	return(words)

    
def document_to_sentences( review, tokenizer, remove_stopwords=False ):
	'''
		Takes a document and inputs it to a list of lists.
	'''
	raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
	sentences = []
	for raw_sentence in raw_sentences:
	    if len(raw_sentence) > 0:
	        sentences.append(document_to_wordlist( raw_sentence, \
	          remove_stopwords ))
	return sentences,raw_sentences

def load_word2vec(dir):
	word2vec = {}
	for path in os.listdir(dir):
		iword2vec = {}
		#load the word2vec features.
		with open(os.path.join(dir,path), 'r') as fin:
			if path == 'vectors0.txt':
				next(fin) #skip information on first line
			for line in fin:
				items = line.replace('\r','').replace('\n','').split(' ')
				if len(items) < 10: continue
				word = items[0]
				vect = np.array([float(i) for i in items[1:] if len(i) > 1])
				iword2vec[word] = vect
		
		word2vec.update(iword2vec)
		
	return word2vec

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

	ALGORITHM = request.form['algorithm']
	LEVEL = request.form['level']
	DOC_COUNT = int(request.form['num-of-docs'])

	DOCUMENTS = []
	for i in range(DOC_COUNT):
		DOCUMENTS.append(request.form['document'+str(i+1)])

	raw_sentences = []

	if LEVEL=="sentence":  
		for each in DOCUMENTS:	# raw sentences will be each document splited into sentences
			raw_sentences+=tokenizer.tokenize(each.decode('utf8').strip())
	else:
		raw_sentences = DOCUMENTS # raw sentence will be the whole do itself.
	matrix = []
	if ALGORITHM=="TF-IDF":
	# Need to write functions for each. Wrote for TF-IDF.
		tfidf = TfidfVectorizer().fit_transform(raw_sentences)
		matrix = (tfidf * tfidf.T).A

	# For each algo the Idea is to form a martix of similarities.
	#---------
		#Algo 2
	
	if ALGORITHM=="LSI":
		texts = [] 
		matrix = np.zeros(shape=(len(raw_sentences), len(raw_sentences)))
		for each in raw_sentences:
			texts.append(document_to_wordlist(each))
		
		dictionary = corpora.Dictionary(texts)
		corpus = [dictionary.doc2bow(text) for text in texts]
		lsii = models.LsiModel(corpus)
		
		matrix = np.zeros(shape=(len(raw_sentences), len(raw_sentences)))

		for i in range(len(raw_sentences)):
		    vec = corpus[i]
		    doc = raw_sentences[i]
		    
		    vec_bow = dictionary.doc2bow(doc.lower().split())
		    vec_lsi = lsii[vec_bow]  # convert the query to LSI space

		    index = similarities.MatrixSimilarity(lsii[corpus])
		    sims = index[vec_lsi]  # perform a similarity query against the corpus
		    cosine = list(enumerate(sims))
		    for j in range(len(raw_sentences)):
		        matrix[i][j] = cosine[j][1]


	#---------
		#Algo 3
	#---------
		#Algo 4
		#Got pretrained vectors from GIT. TA repo has ugly code to generate the same.
	if ALGORITHM=="WORD2VEC":
		word_vector = load_word2vec('static\\vectors')
		matrix = []
		for each in range(len(raw_sentences)):
			li=[]
			for each1 in range(len(raw_sentences)):
				li.append(0)
			matrix.append(li)
		for i in range(0,len(raw_sentences)):
			for j in range(0,len(raw_sentences)):
				sen1 = raw_sentences[i]
				sen2 = raw_sentences[j]
				sen1_words = document_to_wordlist(sen1)
				sen2_words = document_to_wordlist(sen2)
				sen1_vectors = []
				for each in sen1_words:
					if each in word_vector:
						sen1_vectors.append(word_vector[each])
				sen1_vector = np.array(sen1_vectors).sum(axis=0)
				sen2_vectors = []
				for each in sen2_words:
					if each in word_vector:
						sen2_vectors.append(word_vector[each])
				sen2_vector = np.array(sen2_vectors).sum(axis=0)
				matrix[i][j] = cosine_similarity(sen1_vector, sen2_vector)[0][0]

	#---------
	#Forming nodes and links for graph.
	#code might as well be same for all algos.
	#Refine note : Think of creating private funcs and moving code.
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
	c = Counter(wordlist)
	wordcloud = []
	for each in c:
	    temp = {}
	    temp["text"] = each
	    temp["size"] = c[each]*20
	    wordcloud.append(temp)
	wordcloud = json.dumps(wordcloud)
	return render_template('visual.html', graph=graph, sentences=raw_sentences, wordcloud=wordcloud)


if __name__ == '__main__':

	app.debug = True
	
	app.run()