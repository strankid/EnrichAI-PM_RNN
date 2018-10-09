#implement a vectorized approach

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K 

import train_model as train_func

MODEL_NAME = 'models/RNN_Final-97-0.833.model'

#heads = []   #enter the name of head inside the list

df = pd.read_csv("final_dataset.csv")
df = train_func.prepare_test(df)

heads = df.head_id.unique()

final_result_0 = []
final_result_1 = []

K.clear_session()

model = load_model(MODEL_NAME)

for head in heads:
	test_X = train_func.preprocess_test(df, head)
	if len(test_X)==0:
		result_0 = 1
		result_1 = 0
	else:
		result = model.predict(test_X)
		result_0 = result[0][0]
		result_1 = result[0][1]

	final_result_0.append(result_0)
	final_result_1.append(result_1)


final_df = pd.DataFrame()
final_df['head'] = heads
final_df['result_0'] = final_result_0
final_df['result_1'] = final_result_1

final_df.to_csv('results_updated.csv', index=False)

print(result)







		

