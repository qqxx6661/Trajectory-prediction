import numpy as np
import matplotlib.pyplot as plt
import math

size = 7
x = np.arange(size)

Person_0 = [74.915, 64.888, 60.784, 60.444, 68.432, 47.166, 7.9]
Person_1 = [63.22, 68.675, 64.884, 56.747, 66.191, 58.276, 38.832]
Person_2 = [77.288, 69.363, 76.11, 71.719, 69.043, 63.492, 52.174]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Prediction delay (frame)', fontsize=20)
plt.ylabel('Accuracy of Prediction (%)', fontsize=20)
plt.bar(x-0.9*width, Person_0, fc='#faa755', width=0.75*width, label='Person 0')
plt.bar(x, Person_1, fc='#6b473c', width=0.75*width, label='Person 1')
plt.bar(x+0.9*width, Person_2, fc='#8a5d19', width=0.75*width, label='Person 2')
plt.xticks(x, (1, 10, 30, 50, 100, 150, 300), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right', fontsize=13)
plt.show()
