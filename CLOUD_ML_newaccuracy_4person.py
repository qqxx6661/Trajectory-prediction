#!/usr/bin/env python3
# coding=utf-8
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import time


def _generate_path(path_list):
    path_stack = []
    for cam_id in path_list:  # 画出每次的实际路线堆栈
        if not path_stack:
            path_stack.append(cam_id)
        else:
            if cam_id != path_stack[-1]:
                path_stack.append(cam_id)
    return path_stack

def _judge_accuracy(predict_array, real_array):
    correct = 0
    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            # print(predict_array[i], real_array[i])
            correct += 1
        # else:
            # print('错误：', predict_array[i], '实际：', real_array[i])
    correct_rate = correct / len(predict_array)
    return correct_rate * 100


def _judge_accuracy_stack(predict_array_list, labels, label_dealy_list, input_frame):
    correct_list = [0 for _ in range(len(label_dealy_list))]
    index_start = label_dealy_list[-1] + input_frame - 1  # 10+45=55-1=54
    for i in range(index_start, len(predict_array_list)):  # 从54到179
        real_stack = _generate_path(labels[i:])
        for j in range(1, len(label_dealy_list) + 1):  # 从1到6个,依次取前1,2,3,4,5,6个
            predict_stack = _generate_path(predict_array_list[i][:j])
            print(i, 'real:', real_stack, 'prediction:', predict_stack)
            correct_list[j-1] += 1  # 预先假设正确加1
            if len(predict_stack) > len(real_stack):  # 预测多走了摄像头
                print('错误')
                correct_list[j - 1] -= 1  # 有一个错误直接减1
                continue
            for n in range(len(predict_stack)):
                if predict_stack[n] != real_stack[n]:
                    print('错误')
                    correct_list[j - 1] -= 1  # 有一个错误直接减1
                    break

    print(correct_list, len(predict_array_list) - index_start)
    for i in range(len(correct_list)):
        correct_list[i] /= len(predict_array_list) - index_start
    print(correct_list)




def _train_model_save(x_inner, y_inner, name):
    print('---------', name, '---------')
    # SVM-linear过慢
    '''
    print("进行SVM-linear训练")
    start = time.time()
    clf_linear = SVC(kernel='linear').fit(x_inner, y_inner)
    joblib.dump(clf_linear, "ML_model/model_linear_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)
    '''
    # print("进行SVM-rbf训练")
    # start = time.time()
    # clf_rbf = SVC().fit(x_inner, y_inner)
    # joblib.dump(clf_rbf, "ML_model/model_rbf_" + name + ".m")
    # end = time.time()
    # print("执行时间:", end - start)

    # print("进行SVM-sigmoid训练")
    # start = time.time()
    # clf_sigmoid = SVC(kernel='sigmoid').fit(x_inner, y_inner)
    # joblib.dump(clf_sigmoid, "ML_model/model_sigmoid_" + name + ".m")
    # end = time.time()
    # print("执行时间:", end - start)

    print("进行决策树训练")
    start = time.time()
    clf = DecisionTreeClassifier(max_depth=5).fit(x_inner, y_inner)
    joblib.dump(clf, "ML_model/model_tree_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    # print("进行神经网络训练")
    # start = time.time()
    # sc = StandardScaler().fit(x_inner)  # 神经网络和逻辑回归需要预处理数据
    # x_inner = sc.transform(x_inner)
    # mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500).fit(x_inner, y_inner)
    # joblib.dump(mlp, "ML_model/model_mlp_" + name + ".m")
    # end = time.time()
    # print("执行时间:", end - start)

    # print("进行逻辑回归训练")
    # start = time.time()
    # log_reg = linear_model.LogisticRegression(C=1e5).fit(x_inner, y_inner)
    # joblib.dump(log_reg, "ML_model/model_logreg_" + name + ".m")
    # end = time.time()
    # print("执行时间:", end - start)
    # print('-----------------')


def train_model(train_file_inner, input_frame_number_inner, input_label_delay_inner):
    for number in input_label_delay_inner:
        print('---------', number, '---------')
        data = []
        labels = []
        max_train_num = 10000
        mode = 6
        for train_file_each in train_file_inner:
            delay = number  # 改为数组后这个需要放这重置delay
            with open(train_file_each) as file:
                for line in file:
                    tokens = line.strip().split(',')
                    # mode4: 筛选出1234摄像头，其余数据不读取
                    if mode == 4:  # 这里已经没用因为cam4单独分出来文件了
                        if tokens[0] not in ['1', '2', '3', '4', '7', '9', '10', '11', '12', '13']:
                            # print('delete:', tokens)
                            continue
                    data.append([tk for tk in tokens[1:]])
                    if delay != 0:  # 推迟label
                        delay -= 1
                        continue
                    labels.append(tokens[0])
            if number:
                data = data[:-number]  # 删去后面几位

            # print(len(data), len(labels), data[0], data[-1])

        if input_frame_number_inner != 1:

            delay_vector = input_frame_number_inner
            temp_vector = []
            temp_data = []
            # 由于上面已经延迟，所以每个输入对应的输出是输入的最后一行后面的标签
            for line_idx in range(len(data)-input_frame_number_inner+1):
                temp_idx = line_idx
                while delay_vector:
                    temp_vector += data[temp_idx]
                    # print('临时为：', temp_vector)
                    temp_idx += 1
                    delay_vector -= 1

                delay_vector = input_frame_number_inner
                temp_data.append(temp_vector)
                temp_vector = []

            data = temp_data
            labels = labels[input_frame_number_inner-1:]


        if len(data) > max_train_num:  # 控制最大读取行数
            data = data[-max_train_num:]
            labels = labels[-max_train_num:]

        print("输入维度为：", len(data[0]))
        x = np.array(data)
        y = np.array(labels)
        print("总data样本数为：", len(x))
        print("总label样本数为：", len(y))

        # 输出所有数据
        # for i, line in enumerate(data):
        #     print(len(line), line, labels[i])

        _train_model_save(x, y, str(number))


def cal_accuracy(test_file_inner, input_frame_number_inner, input_label_delay_inner):
    test_X_result = []
    test_Y = []
    with open(test_file_inner) as file:
        for line in file:
            tokens = line.strip().split(',')
            test_Y.append(tokens[0])
    for __ in range(180):  # 临时180
        test_X_result.append([])
    for number in input_label_delay_inner:
        print('---------', number, '---------')
        data = []
        labels = []
        delay = number
        with open(test_file_inner) as file:
            for line in file:
                tokens = line.strip().split(',')
                data.append([tk for tk in tokens[1:]])
                if delay != 0:  # 推迟label
                    delay -= 1
                    continue
                labels.append(tokens[0])
        if number != 0:
            data = data[:-number]  # 删去后面几位

        if input_frame_number_inner != 1:
            delay_vector = input_frame_number_inner
            temp_vector = []
            temp_data = []
            # 由于上面已经延迟，所以每个输入对应的输出是输入的最后一行后面的标签
            for line_idx in range(len(data)-input_frame_number_inner+1):
                temp_idx = line_idx
                while delay_vector:
                    temp_vector += data[temp_idx]
                    # print('临时为：', temp_vector)
                    temp_idx += 1
                    delay_vector -= 1

                delay_vector = input_frame_number_inner
                temp_data.append(temp_vector)
                temp_vector = []
            data = temp_data

            labels = labels[input_frame_number_inner-1:]

        test_X = np.array(data)
        # test_Y = np.array(labels)

        # print("读取输入样本数为：", len(test_X))
        # print("读取输出样本数为：", len(test_Y))

        '''
        start = time.time()
        clf_linear_global = joblib.load("model_2cam/model_linear_global.m")
        test_X_result = clf_linear_global.predict(test_X)
        # print("linear全局预测准确率：", _judge_accuracy(test_X_result, test_Y_global))
        print("linear全局预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_global))
        end = time.time()
        print("执行时间:", end - start)
        '''
        # start = time.time()
        # clf_rbf_global = joblib.load("ML_model/model_rbf_global.m")
        # test_X_result = clf_rbf_global.predict(test_X)
        # # print(test_X_result)
        # # print(test_Y)
        # print("rbf全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
        # end = time.time()
        # print("执行时间:", end - start)


        # start = time.time()
        # clf_sigmoid_global = joblib.load("ML_model/model_sigmoid_global.m")
        # test_X_result = clf_sigmoid_global.predict(test_X)
        # print("sigmoid全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
        # end = time.time()
        # print("执行时间:", end - start)

        load_name = "ML_model/model_tree_" + str(number) + '.m'
        start = time.time()
        clf_tree_global = joblib.load(load_name)
        test_X_result_temp = clf_tree_global.predict(test_X)
        print(test_X_result_temp)
        end = time.time()
        print("执行时间:", end - start)
        for i, result in enumerate(test_X_result_temp):
            test_X_result[input_frame_number_inner + number - 1 + i].append(result)


        # # LOC和MLP用
        # sc = StandardScaler().fit(test_X)
        # test_X = sc.transform(test_X)
        #
        # start = time.time()
        # clf_logreg_global = joblib.load("ML_model/model_logreg_global.m")
        # test_X_result = clf_logreg_global.predict(test_X)
        # print("logreg全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
        # end = time.time()
        # print("执行时间:", end - start)
        #
        # start = time.time()
        # clf_mlp_global = joblib.load("ML_model/model_mlp_global.m")
        # test_X_result = clf_mlp_global.predict(test_X)
        # print("mlp全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
        # end = time.time()
        # print("执行时间:", end - start)

    for i, res in enumerate(test_X_result):
        print(i, res)
    _judge_accuracy_stack(test_X_result, test_Y, input_label_delay_inner, input_frame_number_inner)

if __name__ == '__main__':
    glo_start = time.time()
    test_file = "gallery/15-36/15-36_person_0_ML.csv"
    # 180
    # train_file = ['gallery/15-36/15-36_person_0_ML.csv', 'gallery/15-36/15-36_person_1_ML.csv',
    #               'gallery/15-36/15-36_person_2_ML.csv', 'gallery/15-36/15-36_person_4_ML.csv']
    # 360
    # train_file = ['gallery/14-12/14-12_person_0_ML.csv', 'gallery/14-12/14-12_person_1_ML.csv',
    #               'gallery/15-36/15-36_person_0_ML.csv', 'gallery/15-36/15-36_person_1_ML.csv',
    #               'gallery/15-36/15-36_person_2_ML.csv', 'gallery/15-36/15-36_person_4_ML.csv']
    # 720
    train_file = ['gallery/14-12/14-12_person_0_ML.csv', 'gallery/14-12/14-12_person_1_ML.csv',
                  'gallery/14-12/14-12_person_2_ML.csv', 'gallery/14-08/14-08_person_0_ML.csv',
                  'gallery/14-08/14-08_person_1_ML.csv', 'gallery/14-08/14-08_person_2_ML.csv',
                  'gallery/14-14/14-14_person_0_ML.csv', 'gallery/14-14/14-14_person_1_ML.csv',
                  'gallery/15-36/15-36_person_0_ML.csv', 'gallery/15-36/15-36_person_1_ML.csv',
                  'gallery/15-36/15-36_person_2_ML.csv', 'gallery/15-36/15-36_person_4_ML.csv']
    # 6480
    # train_file = ['gallery/14-08/14-08_person_0_ML.csv', 'gallery/14-08/14-08_person_1_ML.csv',
    #               'gallery/14-08/14-08_person_2_ML.csv',
    #               'gallery/14-12/14-12_person_0_ML.csv', 'gallery/14-12/14-12_person_1_ML.csv',
    #               'gallery/14-12/14-12_person_2_ML.csv',
    #               'gallery/14-14/14-14_person_0_ML.csv', 'gallery/14-14/14-14_person_1_ML.csv',
    #               'gallery/14-14/14-14_person_2_ML.csv',
    #               'gallery/15-36/15-36_person_4_ML.csv', 'gallery/15-36/15-36_person_1_ML.csv',
    #               'gallery/15-36/15-36_person_2_ML.csv',
    #               'gallery/14-32/14-32_person_0_ML.csv', 'gallery/14-32/14-32_person_1_ML.csv',
    #               'gallery/14-32/14-32_person_2_ML.csv',
    #               'gallery/14-36/14-36_person_0_ML.csv', 'gallery/14-36/14-36_person_1_ML.csv',
    #               'gallery/14-36/14-36_person_2_ML.csv',
    #               'gallery/14-38/14-38_person_0_ML.csv', 'gallery/14-38/14-38_person_1_ML.csv',
    #               'gallery/14-38/14-38_person_2_ML.csv',
    #               'gallery/14-45/14-45_person_0_ML.csv', 'gallery/14-45/14-45_person_1_ML.csv',
    #               'gallery/14-45/14-45_person_2_ML.csv',
    #               'gallery/14-52/14-52_person_0_ML.csv', 'gallery/14-52/14-52_person_1_ML.csv',
    #               'gallery/14-52/14-52_person_2_ML.csv',
    #               'gallery/14-55/14-55_person_0_ML.csv', 'gallery/14-55/14-55_person_1_ML.csv',
    #               'gallery/14-55/14-55_person_2_ML.csv',
    #               'gallery/14-58/14-58_person_0_ML.csv', 'gallery/14-58/14-58_person_1_ML.csv',
    #               'gallery/14-58/14-58_person_2_ML.csv',
    #               'gallery/15-00/15-00_person_0_ML.csv', 'gallery/15-00/15-00_person_1_ML.csv',
    #               'gallery/15-00/15-00_person_2_ML.csv',
    #               ]
    input_frame_number = 3  # 输入学习帧数
    input_label_delay = [1, 3, 9, 15, 30, 45]  # 预测样本和标签差
    train_model(train_file, input_frame_number, input_label_delay)
    cal_accuracy(test_file, input_frame_number, input_label_delay)
    glo_end = time.time()
    print('global', glo_end - glo_start)
