import numpy as np
import matplotlib.pyplot as plt


x = (1, 2, 3, 4)
NodeRank = (0.5764,	0.6816,	0.8233,	0.8827)
DPCR = (0.5764,	0.5119,	0.7021,	0.8642)
CPCR = (0.5764,	0.7459,	0.8233,	0.8931)
BC = (0.5764, 0.7459, 0.8437, 0.8633)
EC = (0.5168, 0.6310, 0.8437, 0.8426)


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
