import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, Embedding, LeakyReLU, Flatten, GlobalAveragePooling1D
from keras.layers import Bidirectional, GRU, Masking, TimeDistributed, Lambda, Activation, dot, multiply, concatenate, LSTM
from keras.layers.merge import concatenate
from keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from sklearn.metrics import f1_score
from pythonrouge.pythonrouge import Pythonrouge
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import regularizers
from load import load_data

################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Or 2, 3, etc. other than 0
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
################################################################################################

print("Loading data.....")
bert_train, bert_val, bert_test, y_train, y_val  = load_data()
print("Data Loading Completed")


bert_train = bert_train.reshape(6936,1,1024)
bert_val = bert_val.reshape(1000,1,1024)
bert_test = bert_test.reshape(12000,1,1024)
bert_embeddings = Input(shape = (1, 1024), name="bert")
gru_units = 250

rnn = Bidirectional(LSTM(gru_units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25), merge_mode='concat')(elmo_embeddings)

avg_pool = GlobalAveragePooling1D()(rnn)
max_pool = GlobalMaxPooling1D()(rnn)

#model I
merge = concatenate([avg_pool, max_pool])

#model II
o1 = dot([avg_pool, max_pool], axes=[1, 1])
a1 = multiply([o1, avg_pool])


##change here for model I/II
output = Dense(1, activation='sigmoid')(merge)

model = Model(inputs=bert_embeddings, outputs=output) 
model.summary()
#plot_model(model, to_file="model.png")

model.compile(optimizer="adam", loss="binary_crossentropy",  metrics=['accuracy']) 

saved_model = "bert.h5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=10)
callbacks_list = [checkpoint, early]


print("Fitting model...")
history = model.fit(bert_train, y_train, 
	                    batch_size=500,
	                    epochs=25, callbacks=callbacks_list, validation_data=(bert_val, y_val)) 


#plotting
import matplotlib.pyplot as plt 

acc = history.history["acc"]
val_acc = history.history["val_acc"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

f1 = plt.figure()
plt.plot(epochs,acc, label="training")
plt.plot(epochs,val_acc, label="validation")
plt.title("accuracy")
plt.legend()
f1.savefig("accuracy.pdf")

f2 = plt.figure()
plt.plot(epochs,loss, label="training")
plt.plot(epochs,val_loss, label="validation")
plt.title("loss")
plt.legend()
f2.savefig("loss.pdf")



# load model
weights = load_model('bert.h5').get_weights()
model.set_weights(weights)
#model.summary()

probability = model.predict([bert_test]) #
print("no problem")
print(probability.shape)
probability = probability.round()

probability = probability.tolist()

fout = open("predictions.txt", "w+")

print(probability[0][0], type(int(probability[0][0])))
for i in range(len(probability)):
	if int(probability[i][0]) == 1:
		fout.write("INFORMATIVE\n")
	else:
		fout.write("UNINFORMATIVE\n")

fout.close()
