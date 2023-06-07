import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




#synthetic_dataset_results = [98, 99, 100]
#alibaba_dataset_results = [82, 87, 90]

'''randomf_1 = [96, 93, 94, 94]
knn_1 = [91, 92, 92, 89]
dtree_1 = [94, 94, 94, 94]


randomf_2 = [96, 95, 95, 95]
knn_2 = [96, 96, 96, 96]
dtree_2 = [94, 94, 94, 94]

randomf_3 = [93, 92, 93, 94]
knn_3 = [71, 68, 54, 43]
dtree_3 = [71, 68, 54, 43]


randomf_4 = [93, 92, 93, 94]
knn_4 = [92, 91, 91, 90]
dtree_4 = [92, 95, 93, 93]'''

randomf_1 = [94, 96, 97, 93]
knn_1 = [91, 96, 97, 92]
dtree_1 = [94, 94, 97, 92]

approach = ["MUD", "MoN(IoT)r", "PingPong", "Testbed"]
barWidth = 0.15



fig = plt.figure(num=None, figsize=(15, 8.5), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 25})
# Set position of bar on X axis
r1 = np.arange(len(approach))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]


# Make the plot
plt.bar(r1, randomf_1, width=barWidth, label='Random Forest', hatch='\\\\', color = "red")

plt.bar(r2, knn_1, width=barWidth, label='k-Nearest Neighbors', hatch='//', color = 'blue')
plt.bar(r3, dtree_1, width=barWidth, label='Decision Tree', hatch='+', color = 'yellow')


plt.ylabel('Identification accuracy(%)', fontsize=25, fontweight='bold')
plt.xlabel("Datasets", fontsize=25, fontweight='bold')
plt.xticks([r + barWidth-0.0 for r in range(len(approach))], approach)
plt.yticks(np.arange(0,101,10))

plt.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, fontsize=25, shadow=False)
#plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("barplot.pdf", bbox_inches='tight')

plt.show()
plt.close(fig)