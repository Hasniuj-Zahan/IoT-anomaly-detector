import time
import pandas as pd
import os
import psutil

start_time = time.time()

data = pd.read_csv('devicestate0.csv')

class History(object):
	def __init__(self, device=None, log=None):
		self.device = []
		self.log = []


	#property
	def addDevice(self, name):
		self.device.append(name)
		self.log.append([])

	def addlog(self, name, log):
		try:
			index = self.device.index(name)
			self.log[index].append(log)

		except Exception:
			print("device doesn't exist")


	#device.setter
	def name(self, new_device):
		self._device = new_device.title()

	#device.deleter
	def name(self):
		del self._device


a = History()
for col in data.columns:
	a.addDevice(col)
#a.addDevice("Thermostat")
#a.addDevice("Light")
print(a.device)
for j in range(14):
	for i in range(len(a.device)):
		x = a.addlog(a.device[i], data.values[j][i])

a.addlog(a.device[0], 'n4')
a.addlog(a.device[1], 'p1')
a.addlog(a.device[1], 'n3')
print(a.log)
b=0

for i in range(2):
	for b in range(15):
		print(a.log[i][b])
		if a.log[i][b] == 'n4':
			y=(a.log[i][b])
			print("value of y:", y)

			z=(a.device[i])
			print('affected log:', y)

#a.log.remove(y)
nodeName = ['p1', 'n1', 'n2', 'n3', 'n4']

edges = [
    [(''), ('n1')],
    [('p1'), ('n2', 'n3')],
    [('n1'), ('n4')],
    [('n1'), ('n4')],
    [('n2', 'n3'), ('')]
]

#print(edges[1][1][0])


x = "n2"
affected_node_list = []
affected_node_list.append(x)
current_proc_node = 0

node_seq_index = nodeName.index(x)
child = edges[node_seq_index][1]

while current_proc_node != len(affected_node_list):
	x = affected_node_list[current_proc_node]
	node_seq_index = nodeName.index(x)
	child = edges[node_seq_index][1]

	if isinstance(child, tuple):
		for _child in child:
			if _child != '':
				if _child not in affected_node_list:
					affected_node_list.append(_child)

	else:
		if child != '':
			affected_node_list.append(child)

	current_proc_node += 1

j=0
x=(affected_node_list)
print(x)

for i in range(2):
	if y == x[i]:
		print(y)

for sublst in a.log:
	try:
		sublst.remove(y)
	except ValueError:
		pass

print(a.log[0][-1])
#print (a.device)

print("---%s seconds ---" % (time.time() -start_time))

print('The CPU usage is: ', psutil.cpu_percent(1))

pid = os.getpid()
#print(pid)
ps = psutil.Process(pid)

memoryUse = ps.memory_info()
print(memoryUse)
