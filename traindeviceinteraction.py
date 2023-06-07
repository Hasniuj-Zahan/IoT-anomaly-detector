import time
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import seaborn as sn
from sklearn.metrics import confusion_matrix
import csv
import psutil

start_time = time.time()

df = pd.read_csv('devicestate0.csv')
print(df.head())
#df = df.drop(columns=['SourceIP', 'DestinationIP', 'Device_name'])
df = df.dropna()
#changing non numerical data to numerical data

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {'TRUE': 1, 'FALSE': 0}
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
#print(df.head())
#start_time = time.time()

#Model Train
data = pd.read_csv('devicestate0.csv')
data = shuffle(data)
X= df[['Current Temp (F)', 'AC', 'Fan']]
#X = df[['KettleTemp', 'Kettle']]
#X = df[['MotionDetector', 'Livingroomlight', 'LivingroomPlug1', 'Current Temp (F)']]
#X = df[['Length']]
y = df['Label']
#y = df['FlagValue']

print(df.head(5))
print(df.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

models = {}

# Random Forest

#models['Random Forest'] = RandomForestClassifier()
from sklearn.neighbors import KNeighborsClassifier

models['KNeighbors'] = KNeighborsClassifier()

accuracy, precision, recall = {}, {}, {}
for key in models.keys():
    print(key)
    # Fit the classifier model
    if os.path.exists('./' + key.replace(" ", '') + '_' + 'deviceinteraction' + '.pkl'):
        os.remove('./' + key.replace(" ", '') + '_' + 'deviceinteraction' + '.pkl')
    trained_model = models[key].fit(X_train, y_train)
    with open(key.replace(" ", '') + '_' + 'deviceinteraction' + '.pkl', 'wb') as f:
        pickle.dump(trained_model, f)


#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class_labels = {0: "anomaly", 1: "normal"}

#clf = RandomForestClassifier(n_estimators=5)

#from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors =5)

'''from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=100)

# Support Vector Machines
from sklearn.svm import LinearSVC
clf = LinearSVC()

# Decision Trees
clf = DecisionTreeClassifier()'''

clf = clf.fit(X_train,y_train)

#Accuracy Score Calculation
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on testing set: {:.2f}'
     .format(clf.score(X_test, y_test)))

#confidence score
y_predict = (clf.predict_proba(X_test))
with open('confidence1.txt', 'w') as a:
    a.write(str(y_predict))

    #print ('confidence score:', (y_predict))

#Confusion Matrix
y_pred = clf.predict(X_test)
confusion_matrix =confusion_matrix(y_test, y_pred)
print("---% seconds ---" % (time.time() -start_time))
plt.figure(figsize=(6,6))
sn.heatmap(confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

import os
import psutil

pid = os.getpid()
print(pid)
ps = psutil.Process(pid)

memoryUse = ps.memory_info()
print(memoryUse)