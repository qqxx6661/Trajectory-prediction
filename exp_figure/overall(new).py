import numpy as np
import matplotlib.pyplot as plt
import math


def autolabel_user_1(rects):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if i == 0:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.08 * height, "%s" % round(height, 3), ha='center')
        elif i == 1:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, "%s" % round(height, 3), ha='center')
        elif i == 2:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.03 * height, "%s" % round(height, 3), ha='center')
        else:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')


def autolabel_user_2(rects):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if i == 1:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.15 * height, "%s" % round(height, 3), ha='center')
        elif i == 2:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.15 * height, "%s" % round(height, 3), ha='center')
        elif i == 3:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.15 * height, "%s" % round(height, 3), ha='center')
        else:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.15 * height, "%s" % round(height, 3), ha='center')


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
cloud = [(x + y + z) / 180 for x, y, z in zip(transmission_cloud, video_file_cloud, prediction_EaOP)]
print(cloud)
# cloud = log(cloud)

transmission_EaOP = [0.061, 0.135, 0.192, 0.280]
computation_EaOP = [60.837, 60.837, 60.837, 60.837]  # 一台edge（s）
EaOP = [(x + y + z) / 180 for x, y, z in zip(transmission_EaOP, computation_EaOP, prediction_EaOP)]
print(EaOP)
# EaOP = log(EaOP)

error = [0.03, 0.032, 0.034, 0.038]  # 生成一个包含有n个值，均为0.004的list，表示允许的误差范围[-0.003,0.003]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=16)
plt.ylabel('Average Overall Cost (s)', fontsize=16)

rect1 = plt.bar(x - 0.45 * width, cloud, fc='#00A8A8', edgecolor="k", hatch="\\\\\\", yerr=error, width=0.75 * width,
                capsize=8, label='Cloud', zorder=1.8)
rect2 = plt.bar(x + 0.45 * width, EaOP, fc='#730000', edgecolor="k", hatch="xxx", yerr=error, width=0.75 * width,
                capsize=8, label='EATP', zorder=1.8)

plt.xticks(x, (2, 4, 6, 8), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(axis="y", zorder=0.5)  # 生成网格,zorder越小代表越底层，bar设置为1.8刚好不遮住误差线

autolabel_user_1(rect1)
autolabel_user_2(rect2)
plt.show()
