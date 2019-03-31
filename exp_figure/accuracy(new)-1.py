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
