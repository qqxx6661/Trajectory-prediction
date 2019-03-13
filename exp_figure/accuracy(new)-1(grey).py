import numpy as np
import matplotlib.pyplot as plt
import math

size = 6
x = np.arange(size)

Learn_180 = [99.6, 98.85, 91.55, 90.775, 79.5, 78.35]

Learn_360 = [99.8, 98.85, 97.375, 95.85, 90.025, 90.025]

Person_720 = [99.8, 98.85, 98.85, 98.675, 90.425, 90.225]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Prediction range $d_{predict}$ (s)', fontsize=20)
plt.ylabel('Average accuracy (%)', fontsize=20)
plt.bar(x-0.9*width, Learn_180, color="w", edgecolor="k", hatch="\\\\\\", width=0.75*width, label='180s training model')
plt.bar(x, Learn_360, color="w", edgecolor="k", hatch="----", width=0.75*width, label='360s training model')
plt.bar(x+0.9*width, Person_720, color="w", edgecolor="k", hatch=".....", width=0.75*width, label='720s training model')
plt.xticks(x, ('1/3', 1, 3, 5, 10, 15), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.show()
