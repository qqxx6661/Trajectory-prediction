#!/usr/bin/env python3
# coding=utf-8
from scipy.spatial import distance as dist
import numpy as np
import cv2
def reID(filename, gallery_person_list):
    person = 0
    target_hist = np.load(filename)

    # 越大越好cv
    max_similiar = 0
    for i in range(len(gallery_person_list)):
        gallery_hist = np.load(gallery_person_list[i])
        # print(target_hist)
        similiar = cv2.compareHist(target_hist, gallery_hist, cv2.HISTCMP_INTERSECT)
        print(filename, gallery_person_list[i], similiar)
        if similiar == 0.0:  # 写入失败文件忽略
            print('图像数据损坏，不予处理')
            return -1
        if max_similiar < similiar:
            max_similiar = similiar
            person = i

    # # 越小越好cv
    # min_similiar = 99999
    # for i in range(len(gallery_person_list)):
    #     gallery_hist = np.load(gallery_person_list[i])
    #     # print(target_hist)
    #     similiar = cv2.compareHist(target_hist, gallery_hist, cv2.HISTCMP_CHISQR)
    #     print(filename, gallery_person_list[i], similiar)
    #     if similiar == 0.0:  # 写入失败文件忽略
    #         print('图像数据损坏，不予处理')
    #         return -1
    #     if min_similiar > similiar:
    #         min_similiar = similiar
    #         person = i

    # 越小越好distance：cityblock, chebyshev
    # min_similiar = 99999
    # for i in range(len(gallery_person_list)):
    #     gallery_hist = np.load(gallery_person_list[i])
    #     # print(len(target_hist))
    #     similiar = dist.chebyshev(gallery_hist, target_hist)
    #     print(filename, gallery_person_list[i], similiar)
    #     if similiar == 0.0:  # 写入失败文件忽略
    #         print('图像数据损坏，不予处理')
    #         return -1
    #     if min_similiar > similiar:
    #         min_similiar = similiar
    #         person = i

    return person


# reID('EDGE/BSU_data/0/61_0.npy', ['gallery/yolo_1547_61_0.npy', 'gallery/yolo_1547_143_0.npy'])

# print(np.load('BSU_data/0/95_0.npy'))