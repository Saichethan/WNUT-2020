import os
import csv
import numpy as np
import pandas as pd
import pickle




def load_data():
	albert_train = pickle.load(open('embeddings/bert_train.np', 'rb'))
	albert_val = pickle.load(open('embeddings/bert_val.np', 'rb'))
	albert_test = pickle.load(open('embeddings/bert_test.np', 'rb'))
	elmo_train = pickle.load(open('embeddings/elmo_train.np', 'rb'))
	elmo_val = pickle.load(open('embeddings/elmo_val.np', 'rb'))
	xlnet_train = pickle.load(open('embeddings/xlnet_train.np', 'rb'))
	xlnet_val = pickle.load(open('embeddings/xlnet_val.np', 'rb'))

	y_train = pickle.load(open('embeddings/labels_train.np', 'rb'))
	y_val = pickle.load(open('embeddings/labels_val.np', 'rb'))		

	
	return bert_train, bert_val, bert_test, y_train, y_val 


