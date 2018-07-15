import numpy as np
import matplotlib.pyplot as plt


x = (1, 3, 5)
NodeRank = (0.5, 0.7015, 0.8364)
DPCR = (0.3529, 0.679, 0.7928)
CPCR = (0.5, 0.5723, 0.7235)
CC = (0.1985, 0.5658, 0.6940)

plt.plot(x, NodeRank, marker='o', c='r', label='NodeRank')
plt.plot(x, DPCR, marker='>', c='b', label='DPCR')
plt.plot(x, CPCR, marker='^', label='CPCR')
plt.plot(x, CC, marker='x', label='CC')
plt.xticks(x, (1, 3, 5))
plt.xlabel('The number of nodes removed', fontsize=13)
plt.ylabel('Decline rate of network efficiency (%)', fontsize=13)
# plt.legend(loc="lower left", bbox_to_anchor=(0.35, 0.01))
plt.legend()
plt.grid(linestyle='--')
plt.show()
