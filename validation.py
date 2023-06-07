from tkinter import *
from tkinter import filedialog
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import psutil
import pickle
import csv
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix

start_time = time.time()
df = pd.read_csv('pptp.csv')
df = df.drop(columns=['SourceIP', 'DestinationIP'])
df = df.dropna()
print(df.head())

#changing non numerical data to numerical data

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

f = handle_non_numerical_data(df)
print(df.head())



#Model Train

X = df[['DestinationPort', 'Length', 'FlagValue']]
#X = df[['Length']]
#y = df['Label']
#y = df['FlagValue']
X = X.values
print(X)

class_labels = {0: "anomaly", 1: "huebulb" , 2: "wemoinsight", 3: "netgeararlo", 4:"tplinkswitch", 5: "tplinkbulb", 6: "dlinkswitch" , 7:"wemoswitch", 8:"dlinkalarm", 9:"tplinkplug",10: "lifxbulb", 11: "ringalarm", 15: "interaction"} #pingpong
#class_labels = {0: "anomaly", 1: "Plug1", 2: "plug2", 3: "bulb1", 4:"Plug3", 5:"Multiplug", 6:"kettle", 7: 'thermostat', 8: 'bulb2', 9:'plugwm' }
#class_labels = {0: "anomaly", 1: "bulb1" , 2: "brewer", 3: "firetv", 4:"kettle", 5: "Sthub", 6: "echodot" , 7:"wemoplug", 8:"tplinkbulb", 9:"tplinkplug",10: "dryer", 11: "philipbulb", 12: "nest"} #moniot
#fig = plt.figure()
#ax = fig.add_subplot(111)
b = np.random.randint(0, 1000, size=20)
print(b)

with open('RandomForest'+ '_' + 'profiles'+'pingpong' + '.pkl', 'rb') as f:
    clf = pickle.load(f)

x = 0
i = len(X)
print(i)
l=0
warnings.filterwarnings("ignore")

prediction_time = time.time()
prediction = clf.predict(X)
total_time = time.time() - start_time
print(prediction, total_time)



#with open("predictionresult.csv", 'a') as f:
        #    f.write(" : in : " + " : prediction : " + class_labels[prediction[0]]+'\n')
#print("---%s seconds ---" % (time.time() - start_time))
print("---%s seconds ---" % (time.time()- prediction_time))
print('The CPU usage is: ', psutil.cpu_percent(1))
#print("---% seconds ---" % (time.time() - start_time))
#print(" : in : " + " : prediction : " + class_labels[prediction[0]])


import os
import psutil

pid = os.getpid()
print(pid)
ps = psutil.Process(pid)

memoryUse = ps.memory_info()
print(memoryUse)