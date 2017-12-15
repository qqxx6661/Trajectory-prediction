#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import cv2
import os
import time
import numpy as np
import csv
from keras import backend as K
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_eval, yolo_head


class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.6
        self.iou = 0.5

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path)

        # Verify model, anchors, and classes are compatible
        num_classes = len(self.class_names)
        num_anchors = len(self.anchors)
        # TODO: Assumes dim ordering is channel last
        model_output_channels = self.yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Check if model is fully convolutional, assuming channel last order.
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]
        self.is_fixed_size = self.model_image_size != (None, None)

        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(yolo_outputs, self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):

        person_info = []  # 存储person信息

        start = time.time()
        y, x, _ = image.shape
        if self.is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = cv2.resize(image, tuple(reversed(self.model_image_size)), interpolation=cv2.INTER_CUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            image_data = np.array(image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            if predicted_class == 'person' and score >= 0.6:

                label = '{} {:.2f}'.format(predicted_class, score)
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(y, np.floor(bottom + 0.5).astype('int32'))
                right = min(x, np.floor(right + 0.5).astype('int32'))
                # print(label, (left, top), (right, bottom))
                person_info.append([left, top, right, bottom])  # 每有一个人就加一个信息
                cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(image, label, (left, int(top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        end = time.time()
        print('该帧处理时间', end - start)
        return image, person_info

    def close_session(self):
        self.sess.close()


def detect_video(video_name, yolo):
    camera = cv2.VideoCapture(video_name)
    # cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

    # create CSV
    row = []
    timestamp = 0
    dir_name = video_name[-14:-9]
    cam_id = video_name[-5]
    if not os.path.exists('BSU_data/' + dir_name):
        os.makedirs('BSU_data/' + dir_name)
        os.makedirs('BSU_data/' + dir_name + '/' + '0')
        os.makedirs('BSU_data/' + dir_name + '/' + '1')
        os.makedirs('BSU_data/' + dir_name + '/' + '2')
        os.makedirs('BSU_data/' + dir_name + '/' + '3')
        os.makedirs('BSU_data/' + dir_name + '/' + '4')
        os.makedirs('BSU_data/' + dir_name + '/' + '5')
    with open('BSU_data/' + dir_name + '/' + cam_id + '.csv', 'w', newline='') as file:  # newline不多空行
        f = csv.writer(file)

        while True:
            res, frame = camera.read()
            if not res:
                break

            frame_waitcut = frame  # 临时存储frame，保证不要出蓝边
            person_count = 0
            row.append(int(cam_id))
            row.append(timestamp)

            image_withbox, person_info = yolo.detect_image(frame)
            print(person_info)
            for each_info in person_info:

                # 剪切 startx,y endx,y 对应0123
                # 未裁剪person_image
                # person_image = frame_waitcut[each_info[1]:each_info[3], each_info[0]:each_info[2]]
                # 将就下去除蓝色边框
                # person_image = frame_waitcut[each_info[1]+10:each_info[3]-10, each_info[0]+10:each_info[2]-10]

                startX_new = int(each_info[0] + 0.25 * (each_info[2] - each_info[0]))  # 左边剪切25%
                endX_new = int(each_info[2] - 0.25 * (each_info[2] - each_info[0]))  # 右边剪切25%
                startY_new = int(each_info[1] + 0.25 * (each_info[3] - each_info[1]))  # 上边剪切25%
                endY_new = int(each_info[1] + 0.5 * (each_info[3] - each_info[1]))  # 下边剪切50%
                person_image = frame_waitcut[startY_new:endY_new, startX_new:endX_new]

                # display the real-time image
                # try:
                #     cv2.imshow('image', image_person)
                #     # cv2.waitKey(0)
                # except:
                #     continue
                # save as nparray
                reID_feature = reID_extractor(person_image)  # class 'numpy.ndarray'
                reID_filename = 'BSU_data/' + dir_name + '/' + cam_id + '/' + str(timestamp) + '_' + str(person_count) + '.npy'
                np.save(reID_filename, reID_feature)
                person_count += 1
                # save as picture
                cv2.imwrite('reID_image_test/' + cam_id + '_image' + str(timestamp) + '.jpg',
                            person_image)

                row.append([[each_info[0], each_info[1], each_info[2], each_info[3]], reID_filename])

            # cv2.imshow("detection", image_withbox)
            # if cv2.waitKey(1) & 0xff == 27:
            #        break

            print(row)
            print('-----------')
            f.writerow(row)
            row = []
            timestamp += 1

        yolo.close_session()


def detect_img(img, yolo):
    image = cv2.imread(img)
    r_image = yolo.detect_image(image)
    cv2.namedWindow("detection")
    while True:
        cv2.imshow("detection", r_image)
        if cv2.waitKey(110) & 0xff == 27:
                break
    yolo.close_session()


def reID_extractor(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    print(type(hist))
    return hist


if __name__ == '__main__':
    yolo = YOLO()
    # img = '1_image191.jpg'
    video_name = 'video/2017-12-09 15-36-30_5.avi'
    # detect_img(img, yolo)
    start = time.time()
    detect_video(video_name, yolo)
    end = time.time()
    print('all time cost:', end-start)
