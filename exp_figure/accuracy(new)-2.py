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

plt.xlabel('Prediction range $d_{predict}$ (s)', fontsize=16)
plt.ylabel('Average accuracy (%)', fontsize=16)
plt.bar(x-0.9*width, Learn_180, fc='#FFA245', width=0.75*width, label='180s training model', zorder=1.8)
plt.bar(x, Learn_360, fc='#FF4D00', width=0.75*width, label='360s training model', zorder=1.8)
plt.bar(x+0.9*width, Person_720, fc='#600000', width=0.75*width, label='720s training model', zorder=1.8)
plt.xticks(x, ('1/3', 1, 3, 5, 10, 15), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(axis="y", zorder=0.5)  # 生成网格,zorder越小代表越底层，bar设置为1.8刚好不遮住误差线
plt.show()
