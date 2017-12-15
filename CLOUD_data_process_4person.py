#!/usr/bin/env python3
# coding=utf-8
import CLOUD_reID
import csv

exp_info = '15-36'
gallery_person_list = ['gallery/' + exp_info + '/' + '124_0.npy',
                       'gallery/' + exp_info + '/' + '17_0.npy',
                       'gallery/' + exp_info + '/' + '46_0.npy',
                       'gallery/' + exp_info + '/' + '133_0.npy',
                       'gallery/' + exp_info + '/' + '28_0.npy']  # 画廊人员信息(yzd,ck,hyt,sn)
person_track_list = [[], [], [], [], []]
new_person_count = 5  # 5次连续出现，还没使用
# 创建所有帧数组
all_data = []
for frame in range(180):
    all_data.append([frame, ])

# 从6个csv中循环读取数据，组成当前帧所有数据
for filename in range(6):
    with open('EDGE/BSU_data/' + exp_info + '/' + str(filename) + '.csv', "r") as csvFile:
        print('读取:', filename)
        reader = csv.reader(csvFile)
        for frame, item in enumerate(reader):
            if len(item) > 2:
                for person in range(len(item)-2):
                    person_list = eval(item[person+2])  # 转为数组，并在数组首位加上摄像头编号
                    person_list.insert(0, filename)
                    all_data[frame].append(person_list)


# 遍历整个数据集，like：[1778, [5, [37, 81, 108, 234], 'BSU_data/5/1779_0.npy']]
for nei in all_data:
    print(nei)

# 在每一帧的循环中，将该帧出现的人与画廊进行匹配，若与某人相似度大于某值，则认为是那个人
for frame in range(len(all_data)):
    if len(all_data[frame]) > 1:
        for person in range(len(all_data[frame])-1):
            reID_result = CLOUD_reID.reID('EDGE/' + all_data[frame][person+1][2], gallery_person_list)
            if reID_result == -1:  # 写入失败文件忽略
                continue
            print(frame, 'cam_id:', all_data[frame][person+1][0], 'result:', reID_result)
            person_track_list[reID_result].append([frame, all_data[frame][person+1][0],
                                                   all_data[frame][person+1][1]])

# 写入那个人的person_id.csv,[#frame, #camid, #[xywh]]
for gallery_person in range(len(person_track_list)):
    with open('gallery/' + exp_info + '/' + exp_info + '_person_' + str(gallery_person) + '.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(person_track_list[gallery_person])


# 若没有匹配或者画廊里还没有，则加入画廊，给一个新的person_ID

