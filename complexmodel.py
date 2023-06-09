import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil

df = pd.read_csv('MonIoT/MonIoT.csv')
#df.shape

#df = pd.read_csv('pptp.csv')
#df.shape

#df = pd.read_csv('testbedtt.csv')
df.shape

df = df.dropna()

df.shape

df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {'0x0010': 10, '0x0018': 18, '0x0012': 12, '0x0014': 14, '0x0011': 11, '0x0002': 2}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            #print('unique:', unique_elements)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
                print(x)

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

# Splitting the dataset
#df = df.drop(['Column1', ], axis = 1)
df = df.drop(['DestinationPort', ], axis = 1)
df = df.drop(['DestinationIP', ], axis = 1)
df = df.drop(['SourceIP', ], axis = 1)
print(df.shape)

# Target variable and train set
y = df[['Label']]
X = df.drop(['Label', ], axis = 1)

sc = MinMaxScaler()
X = sc.fit_transform(X)

# Split test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

df.head()

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Flatten, BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import *
from keras.layers import *
from keras.activations import *
import tensorflow as tf
from keras import backend as k
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from numpy.random import randn, randint, normal, uniform
import os
from keras.optimizers import Adam
from sklearn.utils import shuffle


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import layers
from tensorflow import keras

#ANN

def fun():
    model = Sequential()
    
    #here 30 is output dimension
    model.add(Dense(32,input_dim =4,activation = 'relu',kernel_initializer='random_uniform'))
    
    #in next layer we do not specify the input_dim as the model is sequential so output of previous layer is input to next layer
    model.add(Dense(16,activation='sigmoid',kernel_initializer='random_uniform'))
    
    #5 classes-normal,dos,probe,r2l,u2r
    model.add(Dense(12,activation='softmax'))
    
    #loss is categorical_crossentropy which specifies that we have multiple classes
    
    model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    return model

model4 = KerasClassifier(build_fn=fun,epochs=50,batch_size=64)

start = time.time()
model4.fit(X_train, y_train.values.ravel())
end = time.time()

print('The CPU usage is: ', psutil.cpu_percent())

pid = os.getpid()
print(pid)
ps = psutil.Process(pid)

memoryUse = ps.memory_info()
print(memoryUse)

print('Training time')


start_time = time.time()
Y_test_pred4 = model4.predict(X_test)
end_time = time.time()

print("Testing time: ",end_time-start_time)

start_time = time.time()
Y_train_pred4 = model4.predict(X_train)
end_time = time.time()

from sklearn.metrics import accuracy_score, precision_score
accuracy_score(y_train, Y_train_pred4)

accuracy_score(y_test, Y_test_pred4)

from keras.metrics.metrics import FalsePositives
from sklearn.metrics import classification_report

print(classification_report(y_test, Y_test_pred4))

def RNN():
  optimize = tf.keras.optimizers.Adam(learning_rate=0.001)
  model = keras.Sequential()
  model.add(layers.Embedding(input_dim=4, output_dim=13))

  # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
  model.add(layers.GRU(32, return_sequences=True))

  # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
  model.add(layers.SimpleRNN(16))

  model.add(layers.Dense(12))
  model.compile(loss ='categorical_crossentropy', optimizer = optimize, metrics = ['accuracy'])

  return model


model5 = KerasClassifier(build_fn=RNN,epochs=50,batch_size=64)

start = time.time()
model5.fit(X_train, y_train.values.ravel())
end = time.time()
print('The CPU usage is: ', psutil.cpu_percent())

pid = os.getpid()
print(pid)
ps = psutil.Process(pid)

memoryUse = ps.memory_info()
print(memoryUse)
print('Training time')
print((end-start))

start_time = time.time()
Y_train_pred5 = model5.predict(X_train)
end_time = time.time()

start_time = time.time()
Y_test_pred5 = model5.predict(X_test)
end_time = time.time()

accuracy_score(y_train, Y_train_pred5)

accuracy_score(y_test,Y_test_pred5)

from sklearn.metrics import classification_report

print(classification_report(y_test, Y_test_pred5))

#CNN implementation 

def CNN():
    optimize= tf.keras.optimizers.Adam(learning_rate=0.001)
    model = Sequential([
    Conv1D(64, 3, padding="same", activation="relu", input_shape=(4,1)),
    MaxPooling1D(pool_size=(2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(12, activation="softmax")
    ])
    model.compile(loss ='sparse_categorical_crossentropy',optimizer = optimize, metrics = ['accuracy'])
    return model

model6 = KerasClassifier(build_fn=CNN,epochs=50,batch_size=64)

start = time.time()
model6.fit(X_train, y_train.values.ravel())
end = time.time()
print('The CPU usage is: ', psutil.cpu_percent())

pid = os.getpid()
print(pid)
ps = psutil.Process(pid)

memoryUse = ps.memory_info()
print(memoryUse)
print('Training time')
print((end-start))

start_time = time.time()
Y_train_pred6 = model6.predict(X_train)
end_time = time.time()

start_time = time.time()
Y_test_pred6 = model6.predict(X_test)
end_time = time.time()


print("Testing time: ",end_time-start_time)

accuracy_score(y_train, Y_train_pred6)

accuracy_score(y_test,Y_test_pred6)

print(classification_report(y_test, Y_test_pred6))

def AE():
    
    input = Input(shape =(4))
    
    #here 30 is output dimension
    x = Dense(16, activation = 'relu', kernel_initializer='random_uniform')(input)
    #x = MaxPooling1D( 3, padding = 'same')(x)
    encoded = Dense(1,activation='relu',kernel_initializer='random_uniform')(x)
    x = Dense(16,activation='relu',kernel_initializer='random_uniform')(encoded)
    decoded = Dense(32,activation='relu',kernel_initializer='random_uniform')(x)
    x = Dense(12,activation='softmax',kernel_initializer='random_uniform')(decoded)
    #x = Dense(30, activation = 'relu', kernel_initializer='random_uniform')(decoded)
    
    #in next layer we do not specify the input_dim as the model is sequential so output of previous layer is input to next layer
    #x = Dense(1,activation='sigmoid',kernel_initializer='random_uniform')(x)
    
    #5 classes-normal,dos,probe,r2l,u2r
    #x = Dense(5, activation='relu')(x)
    #loss is categorical_crossentropy which specifies that we have multiple classes
    model = Model(input, x)
    model.compile(loss ='categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
    
    return model
  
f= AE()

model7 = KerasClassifier(build_fn=AE,epochs=50,batch_size=64)

start = time.time()
model7.fit(X_train, y_train.values.ravel())
end = time.time()
print('The CPU usage is: ', psutil.cpu_percent())

start_time = time.time()
Y_train_pred7 = model7.predict(X_train)
end_time = time.time()

start_time = time.time()
Y_test_pred7 = model7.predict(X_test)
end_time = time.time()

accuracy_score(y_train, Y_train_pred7)

accuracy_score(y_test,Y_test_pred7)

print(classification_report(y_test, Y_test_pred7))
