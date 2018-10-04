#develop metric for True Positives/Negatives

import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint


FUTURE_LENGTH = 1
VALIDATION_HEADS = 12
SEQ_LEN = 10
EPOCHS = 100  
BATCH_SIZE = 10 
NAME = f"{SEQ_LEN}-SEQ-{BATCH_SIZE}-BATCH-{int(time.time())}"



def prepare_train(df):
	df.drop(['Event','Before PM','After PM','end_date','Nozzle-A','Nozzle-B','Nozzle-C','Nozzle-D','Nozzle-E','Nozzle-F','Nozzle-G','Nozzle-H','Nozzle-I','Nozzle-J','Nozzle-K','Nozzle-L','Nozzle-M','Nozzle-N','Nozzle-O','Nozzle-P','Nozzle-Q','Nozzle-R','Nozzle-S','Nozzle-T','Nozzle-U','Nozzle-V','Nozzle-W','Nozzle-X','Nozzle-Y','Nozzle-Z'],axis=1,inplace=True)
	df.start_dt = pd.to_datetime(df.start_dt)
	df.sort_values(['start_dt'],inplace=True)
	df.iloc[:,4:-1] = preprocessing.RobustScaler().fit_transform(df.iloc[:,4:-1])
	df['target'] = df.parent_event.shift(-FUTURE_LENGTH)
	df.dropna(inplace=True)
	df.head

	return df

def prepare_test(df):
	df.drop(['Event','end_date'],axis=1,inplace=True)
	df.start_dt = pd.to_datetime(df.start_dt)
	df.sort_values(['start_dt'],inplace=True)
	df.iloc[:,4:] = preprocessing.RobustScaler().fit_transform(df.iloc[:,4:])
	df.dropna(inplace=True)

	return df

def preprocess_train(df,heads, validation=False):

	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)

	for head in heads:
		data = df[df['head_id']==head]
		data.sort_values(['start_dt'], inplace=True)
		data.drop(['end_dt','head_id','module_position','parent_event'],axis=1,inplace=True) 
		day = data.iloc[0,0].day        
		for i in data.values:
			if((i[0].day<=day+2)|((i[0].day==1)&(day>=30))):
				prev_days.append(i[1:-1])
				if len(prev_days) == SEQ_LEN:
					sequential_data.append([np.array(prev_days), i[-1]])
			else:
				prev_days.clear()
			day = i[0].day
		prev_days.clear()

	random.shuffle(sequential_data)
	#print(pd.DataFrame(sequential_data).shape)

	if validation==False:
		positives = []
		negatives = []

		for seq, target in sequential_data:  
			if target == 0:  
				negatives.append([seq, target])  
			elif target == 1:  
				positives.append([seq, target])  

		random.shuffle(positives)  
		random.shuffle(negatives)  

		lower = min(len(positives), len(negatives)) 

		positives = positives[:lower]  
		negatives = negatives[:2*lower]  
	
		sequential_data = positives+negatives
	
		random.shuffle(sequential_data)
		#print(pd.DataFrame(sequential_data).shape)
	
	X = []
	y = []

	for seq, target in sequential_data:  
		X.append(seq)  
		y.append(target) 

	return np.array(X), y

def preprocess_test(df, heads):

	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)

	for head in heads:
		data = df[df['head_id']==head]
		data.sort_values(['start_dt'], inplace=True)
		data.drop(['end_dt','head_id','module_position'],axis=1,inplace=True) 
		day = data.iloc[0,0].day        
		for i in data.values:
			if((i[0].day<=day+2)|((i[0].day==1)&(day>=30))):
				prev_days.append(i[1:])
				if len(prev_days) == SEQ_LEN:
					sequential_data.append(np.array(prev_days))
			else:
				prev_days.clear()
			day = i[0].day
		prev_days.clear()

	random.shuffle(sequential_data)
	#print(pd.DataFrame(sequential_data).shape)
	
	return np.array(sequential_data)



if __name__ == "__main__":
	df = pd.read_csv("final_model_sep_27.csv")
	df = prepare_train(df)

	heads_train = df.head_id.unique()#[VALIDATION_HEADS:]
	heads_validation = df.head_id.unique()#[:VALIDATION_HEADS]

	train_x, train_y = preprocess_train(df,heads_train)
	validation_x, validation_y = preprocess_train(df,heads_validation, validation=True)



	model = Sequential()
	model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(CuDNNLSTM(128, return_sequences=True))
	model.add(Dropout(0.1))
	model.add(BatchNormalization())

	model.add(CuDNNLSTM(128))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(2, activation='softmax'))


	opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

	# Compile model
	model.compile(
	    loss='sparse_categorical_crossentropy',
	    optimizer=opt,
	    metrics=['accuracy']
	)

	tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

	#filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
	#checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

	# Train model
	history = model.fit(
	    train_x, train_y,
	    batch_size=BATCH_SIZE,
	    epochs=EPOCHS,
	    validation_data=(validation_x, validation_y),
	    callbacks=[tensorboard],#, checkpoint],
	)

	# Score model
	score = model.evaluate(validation_x, validation_y, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	# Save model
	model.save("models/{}.h5".format(NAME))

