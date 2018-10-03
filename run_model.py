import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

import train_model as train_func

MODEL_NAME = 'models/10-SEQ-10-BATCH-1538502212.h5'

#heads = []   #enter the name of head inside the list

df = pd.read_csv("final_model_sep_27.csv")
df = train_func.prepare(df)
heads = df.head_id.unique()[:1]
test_X, test_Y = train_func.preprocess(df, heads, validation=True)

model = load_model(MODEL_NAME)
result = model.predict(test_X[-1:])

print(result)







		

