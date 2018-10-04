import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

import train_model as train_func

MODEL_NAME = 'models/10-SEQ-10-BATCH-1538502212.h5'

#heads = []   #enter the name of head inside the list

df = pd.read_csv("final_dataset.csv")
df = train_func.prepare_test(df)
heads = df.head_id.unique()[:1]
test_X = train_func.preprocess_test(df, heads)

model = load_model(MODEL_NAME)
result = model.predict(test_X)

print(result)







		

