import numpy as np
import matplotlib.pyplot as plt
import math

def autolabel_user_1(rects):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if i == 0:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')
        elif i == 1:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, "%s" % round(height, 3), ha='center')
        elif i == 2:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, "%s" % round(height, 3), ha='center')
        else:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, "%s" % round(height, 3), ha='center')


def autolabel_user_2(rects):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if i == 1:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')
        elif i == 2:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')
        elif i == 3:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')
        else:
            plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, "%s" % round(height, 3), ha='center')

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

video_file_cloud = [86110, 176350, 263380, 395420]  # cloud处理2,4,6,8个摄像头60s（ms）
# 小论文改为6页时要求
# video_file_cloud = [(num / (60 * 3)) for num in video_file_cloud]  # 每帧 除以60*3

# prediction_EaOP = [44.8, 76.16, 116.48, 165.41]  # cloud/edge预测60s视频所用时间，拿到数据后直接预测
prediction_EaOP = [44.8+1800, 76.16+3600, 116.48+5400, 165.41+7200]  # 加上csv读取，写入时间（小论文改为6页时要求）
# 小论文改为6页时要求
# prediction_EaOP = [(num / (60 * 3)) for num in prediction_EaOP]  # 每帧 除以60*3


prediction_cloud = [x + y for x, y in zip(video_file_cloud, prediction_EaOP)]
print("prediction_cloud:", prediction_cloud)
print("prediction_EaOP:", prediction_EaOP)
prediction_cloud = log(prediction_cloud)
prediction_EaOP = log(prediction_EaOP)


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=16)
plt.ylabel('Computational Cost (lg(ms))', fontsize=16)

rect1 = plt.bar(x - 0.45 * width, prediction_cloud, fc='#00A8A8', width=0.75 * width, edgecolor="k", hatch="\\\\\\" , label='Detection+Prediction (Cloud)', capsize=8, zorder=1.8)
rect2 = plt.bar(x + 0.45 * width, prediction_EaOP, fc='#730000', width=0.75 * width, edgecolor="k", hatch="xxx" , label='Prediction (EATP)',capsize=8, zorder=1.8)

plt.xticks(x, (2, 4, 6, 8), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(axis="y", zorder=0.5)  # 生成网格,zorder越小代表越底层，bar设置为1.8刚好不遮住误差线

autolabel_user_1(rect1)
autolabel_user_2(rect2)
plt.show()
