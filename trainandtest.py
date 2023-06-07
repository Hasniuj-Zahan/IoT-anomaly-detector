import time
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import csv
import psutil
from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

start_time = time.time()

#df = pd.read_csv('testbedtt.csv')
#df = pd.read_csv('MonIoT.csv')
df = pd.read_csv('pptp.csv')
print(df.head())
df = df.drop(columns=['SourceIP', 'DestinationIP'])
df = df.dropna()
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
                    x += 1
                print(x)

            df[column] = list(map(convert_to_int, df[column]))

    return df

f = handle_non_numerical_data(df)
#print(df.head())
#start_time = time.time()

#Model Train
#data = pd.read_csv('tplink1.csv')
#data = shuffle(data)
X = df[['DestinationPort', 'Length', 'FlagValue']]
#X = df[['Length']]
y = df['Label']
#y = df['FlagValue']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

models = {}

# Random Forest

#models['Random Forest'] = DecisionTreeClassifier()
models['KNeighbor'] = KNeighborsClassifier()
#models['Gradient Boost'] = GradientBoostingClassifier(random_state=2)

accuracy, precision, recall = {}, {}, {}
for key in models.keys():
    #print(key)
    # Fit the classifier model (MonIoT dataset)
    '''if os.path.exists('./' + key.replace(" ", '') + '_' + 'profiles' + 'testbed' + '.pkl'):
        os.remove('./' + key.replace(" ", '') + '_' + 'profiles' + 'testbed' + '.pkl')
    trained_model = models[key].fit(X_train, y_train)
    with open(key.replace(" ", '') + '_' + 'profiles' + 'testbed' + '.pkl', 'wb') as f:
        pickle.dump(trained_model, f)'''
    # Fit the classifier model (MonIoT dataset)
    if os.path.exists('./' + key.replace(" ", '') + '_' + 'profiles' +'MonIoT'+ '.pkl'):
        os.remove('./' + key.replace(" ", '') + '_' + 'profiles' +'MonIoT'+ '.pkl')
    trained_model = models[key].fit(X_train, y_train)
    with open(key.replace(" ", '') + '_' + 'profiles' + 'MonIoT'+'.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    # Fit the classifier model (pingpong dataset)
    '''if os.path.exists('./' + key.replace(" ", '') + '_' + 'profiles' +'pingpong'+ '.pkl'):
        os.remove('./' + key.replace(" ", '') + '_' + 'profiles' +'pingpong'+ '.pkl')
    trained_model = models[key].fit(X_train, y_train)
    with open(key.replace(" ", '') + '_' + 'profiles' + 'pingpong'+'.pkl', 'wb') as f:
        pickle.dump(trained_model, f)'''


#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#class_labels = {0: "anomaly", "normal": [1, 2, 3, 4, 5, 6, 7]}
#class_labels = {0: "anomaly", 1: "tuyaplug" , 2: "bulb2", 3: "Tplink-Plug", 4:"Tplink-blub", 5: "Multiplug", 6: "kettle" , 7:"thermostat", 8: "wemoplug"} #testbed
#class_labels = {0: "anomaly", 1: "bulb1" , 2: "brewer", 3: "firetv", 4:"kettle", 5: "Sthub", 6: "echodot" , 7:"wemoplug", 8:"tplinkbulb", 9:"tplinkplug",10: "dryer", 11: "philipbulb", 12: "nest"} #moniot
class_labels = {0: "anomaly", 1: "huebulb" , 2: "wemoinsight", 3: "netgeararlo", 4:"tplinkswitch", 5: "tplinkbulb", 6: "dlinkswitch" , 7:"wemoswitch", 8:"dlinkalarm", 9:"tplinkplug",10: "lifxbulb", 11: "ringalarm", 15: "interaction"} #pingpong

#clf = RandomForestClassifier(n_estimators=10)


clf = KNeighborsClassifier(n_neighbors = 5)

# Decision Trees

#clf = tree.DecisionTreeClassifier()

#clf = GradientBoostingClassifier(random_state=2)
'''
# Support Vector Machines
from sklearn.svm import SVC
clf = SVC(kernel="rbf", degree= 6)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=100)

# Support Vector Machines
from sklearn.svm import LinearSVC
clf = LinearSVC()

# Decision Trees
clf = DecisionTreeClassifier()'''

clf = clf.fit(X_train,y_train)
#clf =clf.fit(X_train, y_train.values.ravel())
import os
import psutil

print('The CPU usage is: ', psutil.cpu_percent(1))

pid = os.getpid()
print(pid)
ps = psutil.Process(pid)

memoryUse = ps.memory_info()
print(memoryUse)
Y_test_pred1 = clf.predict(X_test)
#Accuracy Score Calculation
print('Accuracy of Decision Tree classifier on training set: {:.4f}'
      .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on testing set: {:.4f}'
     .format(clf.score(X_test, y_test)))

#confidence score
y_predict = (clf.predict_proba(X_test))

from sklearn.metrics import classification_report

print(classification_report(y_test, Y_test_pred1))

with open('confidence.txt', 'w') as a:
    a.write(str(y_predict))

    #print ('confidence score:', (y_predict))

#Confusion Matrix
'''y_pred = clf.predict(X_test)
confusion_matrix =confusion_matrix(y_test, y_pred)
print("---% seconds ---" % (time.time() -start_time))
plt.figure(figsize=(5,5))
sn.heatmap(confusion_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()'''

