import numpy as np
import matplotlib.pyplot as plt
import math


def autolabel_user_1(rects):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if i == 0:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')
        elif i == 1:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')
        elif i == 2:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')
        else:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')


def autolabel_user_2(rects):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if i == 1:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.03 * height, "%s" % round(height, 3), ha='center')
        elif i == 2:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.03 * height, "%s" % round(height, 3), ha='center')
        elif i == 3:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.03 * height, "%s" % round(height, 3), ha='center')
        else:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.04 * height, "%s" % round(height, 3), ha='center')


def log(list_name):
    for i in range(len(list_name)):
        list_name[i] = math.log10(list_name[i])
        print(list_name[i])
    return list_name


size = 4
x = np.arange(size)

video_file = [11243, 24352, 34260, 45851]  # 视频文件大小60s（KB）
# 小论文改为6页时要求
# video_file = [(num / (60 * 3)) * 1024 for num in video_file]  # 每帧 除以60*3
# print("video_file", video_file)
video_file = log(video_file)

data_to_cloud = [81, 162, 243, 352]  # 所有edge上传的文件大小60s（KB）（2,4,6,8个摄像头）
# 小论文改为6页时要求
# data_to_cloud = [(num / (60 * 3)) * 1024 for num in data_to_cloud]  # 每帧 除以60*3
# print("data_to_cloud", data_to_cloud)
data_to_cloud = log(data_to_cloud)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=16)
plt.ylabel('Communication Cost (lg(KB))', fontsize=16)
rect1 = plt.bar(x - 0.45 * width, video_file, fc='#00A8A8', edgecolor="k", hatch="\\\\\\", width=0.75 * width,
        capsize=8, label='Data to Cloud (Cloud)', zorder=1.8)
# plt.bar(x-0.45*width, data_to_cam, fc='#033649', width=0.75*width, bottom=video_file, label='Feedback (Cloud)')
rect2 = plt.bar(x + 0.45 * width, data_to_cloud, fc='#730000', edgecolor="k", hatch="xxx", width=0.75 * width,
        capsize=8, label='Data to Cloud (EATP)', zorder=1.8)
# plt.bar(x+0.45*width, data_to_cam, fc='#250807', width=0.75*width, bottom=data_to_cloud, label='Feedback (EaOT)')
plt.xticks(x, (2, 4, 6, 8), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(axis="y", zorder=0.5)  # 生成网格,zorder越小代表越底层，bar设置为1.8刚好不遮住误差线
autolabel_user_1(rect1)
autolabel_user_2(rect2)
plt.show()
