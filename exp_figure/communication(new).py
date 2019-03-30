import numpy as np
import matplotlib.pyplot as plt
import math


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

plt.xlabel('Total Camera Numbers', fontsize=18.5)
plt.ylabel('Communication Cost (lg(KB))', fontsize=18.5)
plt.bar(x - 0.45 * width, video_file, fc='#036564', width=0.75 * width, label='Data to Cloud (Cloud)')
# plt.bar(x-0.45*width, data_to_cam, fc='#033649', width=0.75*width, bottom=video_file, label='Feedback (Cloud)')
plt.bar(x + 0.45 * width, data_to_cloud, fc='#764D39', width=0.75 * width, label='Data to Cloud (EATP)')
# plt.bar(x+0.45*width, data_to_cam, fc='#250807', width=0.75*width, bottom=data_to_cloud, label='Feedback (EaOT)')
plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right')
plt.show()
