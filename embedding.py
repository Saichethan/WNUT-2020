import os
import csv
import numpy as np
import pandas as pd
import pickle

from embedding_as_service.text.encode import Encoder  


def load_emb():

	sentences = []
	sentences_test = []
	labels = []
	bertt = []
	bert = []
	elmo = []
	xlnet = []
	train_file = "/home/reddy/WNUT/data/train.tsv"
	val_file = "/home/reddy/WNUT/data/valid.tsv"
	test_file = "/home/reddy/WNUT/data/test.tsv"

	train_file_list = open(train_file,'rt') 
	train_data = csv.reader(train_file_list, delimiter='\t')
	train_data_list = list(train_data)

	val_file_list = open(val_file,'rt') 
	val_data = csv.reader(val_file_list, delimiter='\t')
	val_data_list = list(val_data)

	test_file_list = open(test_file,'rt') 
	test_data = csv.reader(test_file_list, delimiter='\t')
	test_data_list = list(test_data)
	
	train_length = len(train_data_list)
	val_length = len(val_data_list)

	for i in range(1, train_length):
		sentences.append(train_data_list[i][1])
		temp_label = str(train_data_list[i][-1])
		
		if temp_label == "INFORMATIVE":
			labels.append(int(1))
		elif temp_label == "UNINFORMATIVE":
			labels.append(int(0))

	print("Number of Tweets in training: ", train_length, len(labels))


	for i in range(1, val_length):
		sentences.append(val_data_list[i][1])
		temp_label = str(val_data_list[i][-1])
		
		if temp_label == "INFORMATIVE":
			labels.append(int(1))
		elif temp_label == "UNINFORMATIVE":
			labels.append(int(0))

	print("Number of Tweets in validation: ", val_length, len(labels))

	test_length = len(test_data_list)
	for i in range(test_length):
		sentences.append(test_data_list[i][1])
		sentences_test.append(test_data_list[i][1])

	print("Number of sentences: ", len(sentences))

	print("lengths: ", train_length, val_length)
	xl = Encoder(embedding='xlnet', model='xlnet_large_cased')
	xlnet_temp = xl.encode(texts=sentences, pooling='reduce_mean')


	el = Encoder(embedding='elmo', model='elmo_bi_lm')
	elmo_temp = el.encode(texts=sentences, pooling='reduce_mean') 
	
	bt = Encoder(embedding='bert', model='bert_large_cased')
	bert_temp = xl.encode(texts=sentences, pooling='reduce_mean')
	bert_test = xl.encode(texts=sentences_test, pooling='reduce_mean')

	for i in range(len(elmo_temp)):
		elvector = elmo_temp[i]
		elmo.append(elvector)
	
	elmo = np.asarray(elmo)
	

	for i in range(len(xlnet_temp)):
		xlvector = xlnet_temp[i]
		xlnet.append(xlvector)

	xlnet = np.asarray(xlnet)


	for i in range(len(bert_temp)):
		bert_vector = bert_temp[i]
		bert.append(bert_vector)
	
	bert = np.asarray(bert)

	for i in range(len(bert_test)):
		bert_t_vector = bert_test[i]
		bertt.append(bert_t_vector)

	np.ndarray.dump(bertt, open('embeddings/bert_test.np', 'wb'))

	
	np.ndarray.dump(bert[:train_length-1], open('embeddings/bert_train.np', 'wb'))
	np.ndarray.dump(bert[train_length-1 : train_length + val_length-2], open('embeddings/bert_val.np', 'wb'))

	
	np.ndarray.dump(xlnet[:train_length-1], open('embeddings/xlnet_train.np', 'wb'))
	np.ndarray.dump(elmo[:train_length-1], open('embeddings/elmo_train.np', 'wb'))

	np.ndarray.dump(xlnet[train_length-1 : train_length + val_length-2], open('embeddings/xlnet_val.np', 'wb'))
	np.ndarray.dump(elmo[train_length-1 : train_length + val_length-2], open('embeddings/elmo_val.np', 'wb'))
	
	labels = np.asarray(labels)
	np.ndarray.dump(labels[:train_length-1], open('embeddings/labels_train.np', 'wb'))
	np.ndarray.dump(labels[train_length-1 : train_length + val_length-2], open('embeddings/labels_val.np', 'wb'))
	

load_emb()
