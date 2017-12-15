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

transmission_cloud = [625000, 1250000, 1875000, 2500000]  # 2,4,6,8个摄像头每帧 50M=5120KB/S
log(transmission_cloud)

transmission_EaOP = [1289, 2578, 3867, 5156]
log(transmission_EaOP)

video_file_cloud = [2613000, 5152000, 7839000, 10452000]  # cloud处理2,4,6,8个摄像头（30s）
prediction_cloud = [694.75, 771.51, 867.89, 964.32]
computation_cloud = [1256694.75, 2613771.51, 3845867.89, 5152964.32]  # 上面两数组相加
# ave(computation_cloud)
log(computation_cloud)

video_file_EaOP = [1256000, 2613000, 3845000, 5152000]  # edge各自处理1,2,3,4个摄像头（30s）这里暂时用cloud的计算能力，940MX
prediction_EaOP = [694.75, 771.51, 867.89, 964.32]
computation_EaOP = [1256694.75, 2613771.51, 3845867.89, 5152964.32]  # 上面两数组相加
# ave(computation_EaOP)
log(computation_EaOP)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=20)
plt.ylabel('Overall Cost (lg(ms))', fontsize=20)


plt.bar(x-0.45*width, transmission_cloud, fc='#036564', width=0.75*width, label='Transmission (Cloud)')
plt.bar(x-0.45*width, computation_cloud, fc='#033649', width=0.75*width, bottom=transmission_cloud, label='Computation (Cloud)')
plt.bar(x+0.45*width, transmission_EaOP, fc='#764D39', width=0.75*width, label='Transmission (EATP)')
plt.bar(x+0.45*width, computation_EaOP, fc='#250807', width=0.75*width, bottom=transmission_EaOP, label='Computation (EATP)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.show()
