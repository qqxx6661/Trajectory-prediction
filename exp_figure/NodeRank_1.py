import numpy as np
import matplotlib.pyplot as plt


x = (1, 2, 3)
NodeRank = (0.5128,0.7399,0.8708)
DPCR = (0.4615,0.8236,1)
CPCR = (0.5128,0.6298,0.7233)
BC = (0.5128,0.7399,0.9213)
EC = (0.4615,0.6890,0.9213)


plt.plot(x, NodeRank, marker='o', c='r', label='NodeRank')
plt.plot(x, DPCR, marker='>', c='b', label='DPCR')
plt.plot(x, CPCR, marker='^', label='CPCR')
plt.plot(x, BC, marker='x', label='BC')
plt.plot(x, EC, marker='+', label='EC')
plt.xticks(x, (1, 2, 3, 4))
plt.xlabel('The number of nodes removed', fontsize=13)
plt.ylabel('Decline rate of network efficiency (%)', fontsize=13)
# plt.legend(loc="lower left", bbox_to_anchor=(0.35, 0.01))
plt.legend()
plt.grid(linestyle='--')
plt.show()
