#!/usr/bin/env python3
# coding=utf-8
import csv

exp_info = '15-36'
person = 'person_4'  # 文件名
cal_speed_delay = 1  # 连续在同一摄像头n帧后再计算速度
cal_speed_delay_flag = 1  # 连续在同一摄像头n帧都有数据则置为1

def relative_position(cood):
    # 输入两点坐标的list，其实就是矩形的左上角和右下角坐标。求该点在画面内的相对位置（左右来看）。视频分辨率为640x480。
    # 举例：该点x坐标是320，那他的相对位置就是0
    value = ((0.5 * (cood[0] + cood[2])) - 320) / 320
    value = int(value * 100)
    return value

def cam_predict_relative(cam_id, position):  # 由于场景是预先设计好的，所以这里需要手动设置
    cam = [0, 0, 0, 0, 0, 0]
    if cam_id >= 6: return cam
    cam[cam_id] = 100
    if cam_id == 0:
        if position > 0: cam[1] = position
    elif cam_id == 1:
        if position > 0: cam[2] = position
        else: cam[0] = abs(position)
    elif cam_id == 2:
        if position > 0: cam[3] = position
        else: cam[1] = cam[4] = abs(position)
    elif cam_id == 3:
        if position < 0: cam[2] = abs(position)
    elif cam_id == 4:
        if position < 0: cam[5] = abs(position)
        else: cam[2] = abs(position)
    else:
        if position > 0: cam[4] = position
    return cam

def judge_cam_location(curr_line, prev_list):
    # 判断是否关联并且进入正负数是否合理，由于新实验很多是漏检测而非错误，所以comment掉，之后可以改回来
    if prev_list[1] == 0:
        if curr_line[1] == 1 and curr_line[3] < 0 and prev_list[3] > 0: return True
    if prev_list[1] == 1:
        if curr_line[1] == 0: return True
        if curr_line[1] == 2: return True
        if curr_line[1] == 4: return True
    if prev_list[1] == 2:
        if curr_line[1] == 1: return True
        # if curr_line[1] == 1 and curr_line[3] > 0 and prev_list[3] < 0: return True
        if curr_line[1] == 3: return True
        if curr_line[1] == 4: return True
        if curr_line[1] == 5: return True
    if prev_list[1] == 3:
        if curr_line[1] == 2: return True
    if prev_list[1] == 4:
        if curr_line[1] == 2: return True
        # if curr_line[1] == 2 and curr_line[3] > 0 and prev_list[3] < 0: return True
        if curr_line[1] == 5: return True
        # if curr_line[1] == 5 and curr_line[3] < 0 and prev_list[3] > 0: return True
    if prev_list[1] == 5:
        if curr_line[1] == 4: return True
    return False

def cam_generate(pre_cam_id, cur_cam_id):
    if pre_cam_id == cur_cam_id:
        return pre_cam_id
    if pre_cam_id == -1:
        if cur_cam_id == 0:
            return 6
        elif cur_cam_id == 2:
            return 7
        elif cur_cam_id == 3:
            return 7
        else:
            return 8
    elif (pre_cam_id == 0 and cur_cam_id == 1) or (pre_cam_id == 1 and cur_cam_id == 0):
        return 9
    elif (pre_cam_id == 1 and cur_cam_id == 2) or (pre_cam_id == 2 and cur_cam_id == 1):
        return 10
    elif (pre_cam_id == 2 and cur_cam_id == 3) or (pre_cam_id == 3 and cur_cam_id == 2):
        return 11
    elif (pre_cam_id == 2 and cur_cam_id == 4) or (pre_cam_id == 4 and cur_cam_id == 2) \
            or (pre_cam_id == 2 and cur_cam_id == 5) or (pre_cam_id == 5 and cur_cam_id == 2)\
            or (pre_cam_id == 1 and cur_cam_id == 4) or (pre_cam_id == 4 and cur_cam_id == 1):
        return 12
    elif (pre_cam_id == 4 and cur_cam_id == 5) or (pre_cam_id == 5 and cur_cam_id == 4):
        return 13

# 创建所有帧数组
all_data = []
for frame in range(180):
    all_data.append([frame, ])
all_data_ML = []

# 读取
with open('gallery/' + exp_info + '/' + exp_info + '_' + person + '.csv') as csvFile:
    reader = csv.reader(csvFile)
    for item in reader:
        frame_now = int(item[0])  # 当前处理帧

        # # 长度大于1说明这帧之前有了，异常.由于暂时一共就两个人，所以只比较两个值，选择距离比较小的一个
        # try:
        #     # 这里若报错，说明前面一帧也没有信息，假设没有出现这种情况
        #     if all_data[frame_now-1][1] == all_data[frame_now][1]:  # 该帧已有数据和上一阵在同一摄像头内
        #         print('该帧已有数据和上一阵在同一摄像头内')
        #         # 不需要存储了，使用上次数据
        #         continue
        #     elif all_data[frame_now-1][1] == int(item[1]):  # 该帧新数据和上一阵在同一摄像头内
        #         print('该帧新数据和上一阵在同一摄像头内')
        #         while len(all_data[frame_now]) > 1:
        #             all_data[frame_now].pop()
        #     else:  # 摄像头都与上帧不同，采用相对位置，不过这应该没有意义了
        #         last_position = int(all_data[frame_now-1][3])
        #         print('上个位置与已加入', all_data[frame_now][3] - last_position,
        #               '上个位置与新加入', relative_position(eval(item[2])) - last_position)
        #         if abs(all_data[frame_now][3] - last_position) > abs(relative_position(eval(item[2])) - last_position):
        #             while len(all_data[frame_now]) > 1:
        #                 all_data[frame_now].pop()
        #         else:
        #             # 不需要存储了，使用上次数据
        #             continue
        # except IndexError as e:
        #     print(e)

        all_data[frame_now].append(int(item[1]))  # 加入camid
        all_data[frame_now].append(eval(item[2]))  # 加入位置，主要用于速度计算
        all_data[frame_now].append(relative_position(eval(item[2])))  # 加入相对位置

        # 加入速度
        for i in range(cal_speed_delay):
            # print(all_data[frame_now - (i+1)])
            # 连续N帧有
            if len(all_data[frame_now - (i+1)]) == 1:
                # print('前第', i+1, '帧没信息，不计算速度')
                cal_speed_delay_flag = 0
                break
            # 连续N帧在同一个摄像头内
            if all_data[frame_now - (i+1)][1] != int(item[1]):
                # print('前第', i+1, '帧不在同一摄像头内，不计算速度')
                cal_speed_delay_flag = 0
                break
        if cal_speed_delay_flag == 0:
            cal_speed_delay_flag = 1
            # print('跳过该帧速度')
            continue
        # print('加入速度')
        speed_x = int(eval(item[2])[0]) - int(all_data[frame_now-1][2][0])
        speed_y = int(eval(item[2])[1]) - int(all_data[frame_now-1][2][1])
        all_data[frame_now].append([speed_x, speed_y])  # x,y轴速度

# 这里插入优化函数，去除出错的信息，之后再存入person_x_predict中
prev_frame = []
for l, line in enumerate(all_data):
    if len(line) == 1:  # 如果该帧没信息，跳过
        continue
    else:
        if not prev_frame:  # 如果之前没信息，给第一次信息，跳过
            prev_frame = line
        else:
            if line[1] == prev_frame[1]:  # 如果相同摄像头,距离绝对值必须小于50
                if abs(line[3] - prev_frame[3]) <= 180:  # 新实验漏检测，这里限制设置大一些
                    prev_frame = line
                else:
                    print('去除', line, '对比', prev_frame)
                    all_data[l] = line[:1]
            else:  # 如果摄像头不相同，必须满足1.相连摄像头，2.距离一左一右正负必须相反
                if judge_cam_location(line, prev_frame):
                    prev_frame = line
                else:
                    print('去除', line, '对比', prev_frame)
                    all_data[l] = line[:1]

# 先删除没有速度的data
for i, each_data in enumerate(all_data):
    if len(each_data) < 5:
        all_data[i] = all_data[i][:1]

# 遍历all_data
# for line in all_data:
#     print(line)

# 写入person_x_predict
with open('gallery/' + exp_info + '/' + exp_info + '_' + person + '_apredict.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(all_data)


blank_count = 0  # 空白帧数
cam_stack = [-1]
start_loc = 0
end_loc = 0
last_index = 0
for i, each_data in enumerate(all_data):
    if len(each_data) != 5 and blank_count != -2:  # 之前没有,现在还没有
        blank_count += 1
    elif len(each_data) != 5 and blank_count == -2:  # 之前有数据，现在没有了
        start_loc = all_data[i-1][3]
        blank_count = 1
        if all_data[i-1][1] != cam_stack[-1]:  # 说明这段间隔是同一摄像头
            cam_stack.append(all_data[i-1][1])
    elif len(each_data) == 5 and blank_count == -2:  # 在连续数据中间
        last_index = i  # 给末尾补全用
        continue
    else:  # 到了有数据的帧
        cam_stack.append(each_data[1])
        end_loc = each_data[3]
        mid_cam = cam_generate(cam_stack[-2], cam_stack[-1])
        if cam_stack[-2] == cam_stack[-1]:  # 说明在同一摄像头内
            speed_frame = abs(start_loc - end_loc) / blank_count
            loc = start_loc
            if start_loc < end_loc:  # 同摄像头内起始位置比终点位置小，所以升序
                order = 'ascend'
            else:
                order = 'descend'
        elif cam_stack[-2] < cam_stack[-1]:  # 说明从小编号走到大编号
            speed_frame = 100 / blank_count
            loc = -100
            order = 'ascend'
        else:  # 说明从大编号走到小编号
            speed_frame = 100 / blank_count
            loc = 100
            order = 'descend'

        while blank_count:
            all_data[i-blank_count].append(mid_cam)
            all_data[i-blank_count].append([0, 0, 0, 0])
            # print('前：', cam_stack[-2], '后：', cam_stack[-1], '开始坐标：', start_loc, '结束坐标：', end_loc)
            if order == 'descend':
                loc = int(loc - speed_frame)
                all_data[i - blank_count].append(loc)
                all_data[i - blank_count].append([-int(speed_frame), 0])
            if order == 'ascend':
                loc = int(loc + speed_frame)
                all_data[i - blank_count].append(loc)
                all_data[i - blank_count].append([int(speed_frame), 0])
            # print(blank_count, i - blank_count, '改变为', all_data[i - blank_count])
            blank_count -= 1

        # 重置临时变量
        blank_count = -2  # -2代表不是第一次了，-1则是刚开始
        start_loc = 0
        end_loc = 0
        loc = 0
        last_index = i  # 给末尾补全用

print(last_index)
for i in range(last_index+1, len(all_data)):
    all_data[i].append(all_data[last_index][1])
    all_data[i].append(all_data[last_index][2])
    all_data[i].append(all_data[last_index][3])
    all_data[i].append([0, 0])

# 写入person_x_full
with open('gallery/' + exp_info + '/' + exp_info + '_' + person + '_full.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(all_data)

# 写入person_x_ML
for line in all_data:
    if len(line) == 5:  # 有速度的才处理
        ML_temp = []
        ML_temp.append(line[1])
        ML_temp.append(line[0])
        ML_temp.append(line[4][0])
        ML_temp.append(line[4][1])
        cam_list = cam_predict_relative(int(line[1]), line[3])
        for cam_value in cam_list:
            ML_temp.append(cam_value)
        all_data_ML.append(ML_temp)

with open('gallery/' + exp_info + '/' + exp_info + '_' + person + '_ML.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(all_data_ML)
