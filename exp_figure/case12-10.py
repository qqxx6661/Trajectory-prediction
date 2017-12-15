import numpy as np
import matplotlib.pyplot as plt
import math

size = 3
x = np.arange(size)

Case_0 = [25.763, 19.793, 11.943]
Case_1 = [42.712, 40.275, 35.116]


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Prediction delay (frame)', fontsize=20)
plt.ylabel('Accuracy of Prediction (%)', fontsize=20)
plt.bar(x-0.45*width, Case_0, fc='#6b473c', width=0.75*width, label='Person 1')
plt.bar(x+0.45*width, Case_1, fc='#8a5d19', width=0.75*width, label='Person 2')
plt.xticks(x, ('1', '10', '30', '50', '100'), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper right', fontsize=15)
plt.show()
