import numpy as np
import matplotlib.pyplot as plt
import math


def log(list_name):
    for i in range(len(list_name)):
        list_name[i] = math.log10(list_name[i])
        print(list_name[i])
    return list_name

def ave(list_name):
    for i in range(len(list_name)):
        list_name[i] = list_name[i] / 900
        print(list_name[i])
    return list_name

size = 4
x = np.arange(size)

transmission_cloud = [9.980, 17.256, 30.162, 41.513]  # 2,4,6,8 段视频
video_file_cloud = [86.11, 176.35, 263.38, 395.42]  # cloud处理2,4,6,8个摄像头60s（s）
prediction_EaOP = [0.045, 0.076, 0.116, 0.165]  # cloud/edge预测60s视频所用时间，拿到数据后直接预测
cloud = [(x+y+z)/180 for x, y, z in zip(transmission_cloud, video_file_cloud, prediction_EaOP)]
print(cloud)
# cloud = log(cloud)

transmission_EaOP = [0.061, 0.135, 0.192, 0.280]
computation_EaOP = [60.837, 60.837, 60.837, 60.837]  # 一台edge（s）
EaOP = [(x+y+z)/180 for x, y, z in zip(transmission_EaOP, computation_EaOP, prediction_EaOP)]
print(EaOP)
# EaOP = log(EaOP)



total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=19)
plt.ylabel('Average Overall Cost (s)', fontsize=19)


plt.bar(x-0.45*width, cloud, fc='#036564', width=0.75*width, label='Cloud')
# plt.bar(x-0.45*width, computation_cloud, fc='#033649', width=0.75*width, bottom=transmission_cloud, label='Computation (Cloud)')
plt.bar(x+0.45*width, EaOP, fc='#764D39', width=0.75*width, label='EATP')
# plt.bar(x+0.45*width, computation_EaOP, fc='#250807', width=0.75*width, bottom=transmission_EaOP, label='Computation (EATP)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=15)
plt.show()
