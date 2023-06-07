from tkinter import *
from tkinter import filedialog
import pandas as pd
import csv
import os
import time

start_time = time.time()
#converting pcap file to csv
os.system("tshark -r 10action1.pcap -T fields -2 -e frame.time_epoch -e ip.src -e ip.dst -e frame.len -e tcp.port -e tcp.flags  -E separator=, -E occurrence=f > 10action1.csv")

#adding header
file1 = ("C:\\Users\\Asus\\PycharmProjects\\WemoPlug\\10action1.csv")
#file1 = filedialog.askopenfilename(initialdir="C:\\Users\\Asus\\PycharmProjects\\WemoPlug\\multiplug.csv")
with open(file1, newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open(file1,'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Time', 'SourceIP', 'DestinationIP', 'Length','DestinationPort','FlagValue'])
    #w.writerow(['Time', 'SourceIP', 'DestinationIP', 'Length', 'DestinationPort', 'Info'])
    w.writerows(data)
import os
import psutil

pid = os.getpid()
print(pid)
ps = psutil.Process(pid)

memoryUse = ps.memory_info()
print(memoryUse)
print("---% seconds ---" % (time.time() -start_time))

'''
#Adding Device name and label
#file1 = filedialog.askopenfilename(initialdir="C:\\Users\\Asus\\PycharmProjects")
df = pd.read_csv(file1)
print(df.head())
df['Device_name'] = ('Smart Tap')
df['Label']= ('12')
print (df.head())
file2 = df.to_csv(file1)
print("---% seconds ---" % (time.time() -start_time))


#print(file2.head())

file3 = filedialog.askopenfilename(initialdir="C:\\Users\\Asus\\PycharmProjects\\WemoPlug")

print("Merging CSV files...")

# merge Device profile together
dataFrame = pd.concat(map(pd.read_csv, [file1, file3]), ignore_index=True)
print(dataFrame)
dataFrame = dataFrame.drop(columns=['a'])
dataFrame.to_csv('TestTrain.csv')'''



