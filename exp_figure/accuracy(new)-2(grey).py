import numpy as np
import matplotlib.pyplot as plt
import math

size = 6
x = np.arange(size)

Learn_180 = [98.73, 96.27, 78.93, 78.4, 74.43, 73.43]

Learn_360 = [99.2, 97.5, 91.7, 91.7, 80.5, 79.7]

Person_720 = [99.2, 97.8, 97.2, 96.5, 88.5, 87.6]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Prediction range $d_{predict}$ (s)', fontsize=20)
plt.ylabel('Average accuracy (%)', fontsize=20)
plt.bar(x-0.9*width, Learn_180, color="#AFEEEE", edgecolor="k", hatch="\\\\\\", width=0.75*width, label='180s training model')
plt.bar(x, Learn_360, color="#FFA500", edgecolor="k", hatch="----", width=0.75*width, label='360s training model')
plt.bar(x+0.9*width, Person_720, color="#800000", edgecolor="k", hatch=".....", width=0.75*width, label='720s training model')
plt.xticks(x, ('1/3', 1, 3, 5, 10, 15), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.show()
