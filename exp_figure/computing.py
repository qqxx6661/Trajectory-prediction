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

video_file_cloud = [2613000, 5152000, 7839000, 10452000]  # cloud处理2,4,6,8个摄像头（60s）
# video_file_cloud = ave(video_file_cloud)
log(video_file_cloud)


prediction_EaOP = [694.75, 771.51, 867.89, 964.32]  # cloud预测60s视频所用时间，拿到数据后直接预测
# prediction_EaOP = ave(prediction_EaOP)
prediction_EaOP = log(prediction_EaOP)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=19)
plt.ylabel('Computational Cost (lg(ms))', fontsize=19)


plt.bar(x-0.45*width, video_file_cloud, fc='#036564', width=0.75*width, label='Body Detection (Cloud)')
plt.bar(x-0.45*width, prediction_cloud, fc='#033649', width=0.75*width, bottom=video_file_cloud, label='Prediction (Cloud)')
# plt.bar(x+0.45*width, video_file_edge, fc='#764D39', width=0.75*width, label='Object Detection (EaOP)')
plt.bar(x+0.45*width, prediction_EaOP, fc='#250807', width=0.75*width, label='Prediction (EATP)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right', fontsize=14)
plt.show()
