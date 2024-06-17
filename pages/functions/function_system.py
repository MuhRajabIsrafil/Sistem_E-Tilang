import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd


def xyxy_to_xywh(xyxy):
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return np.array([int(x_temp), int(y_temp), int(w_temp), int(h_temp)])


def get_concat_h_resize(im1, im2, resize_big_image=True):
    if im1.shape[0] == im2.shape[0]:
        _im1 = im1
        _im2 = im2
    elif (((im1.shape[0] > im2.shape[0]) and resize_big_image) or
          ((im1.shape[0] < im2.shape[0]) and not resize_big_image)):
        _im1 = cv2.resize(im1, (int(im1.shape[1] * im2.shape[0] / im1.shape[0]), im2.shape[0]))
        _im2 = im2
    else:
        _im1 = im1
        _im2 = cv2.resize(im2, (int(im2.shape[1] * im1.shape[0] / im2.shape[0]), im1.shape[0]))
    dst = cv2.hconcat([_im1, _im2])
    return dst


def get_concat_v_resize(im1, im2, resize_big_image=True):
    if im1.shape[1] == im2.shape[1]:
        _im1 = im1
        _im2 = im2
    elif (((im1.shape[1] > im2.shape[1]) and resize_big_image) or
          ((im1.shape[1] < im2.shape[1]) and not resize_big_image)):
        _im1 = cv2.resize(im1, (im2.shape[1], int(im1.shape[0] * im2.shape[1] / im1.shape[1])))
        _im2 = im2
    else:
        _im1 = im1
        _im2 = cv2.resize(im2, (im1.shape[1], int(im2.shape[0] * im1.shape[1] / im2.shape[1])))
    dst = cv2.vconcat([_im1, _im2])
    return dst


def get_middle_coordinates(bbox, classes):
    x_min, y_min, x_max, y_max = bbox
    y_middle = round((y_min + y_max) / 2)
    if classes == "no_helmet":
        return np.array([x_min, y_min, x_max, y_middle])
    else:
        return np.array([x_min, y_middle, x_max, y_max])


def MakeFolder(folder):
    os.mkdir('../results/' + folder)


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 5, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, (p1[0], p1[1] + 5), p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    5,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)


def get_time(time):
    sec = 1
    date_format_str = '%d/%m/%Y %H:%M:%S'

    time_str = datetime.strptime(time, date_format_str)
    time_change = time_str + pd.DateOffset(seconds=sec)

    final_time = time_change.strftime('%d/%m/%Y %H:%M:%S')
    return final_time


def fix_array(data):
    fix_values = []

    for item in data:
        if item[0] not in fix_values:
            fix_values.append(item[0])

    return fix_values


def Make_BG(size):
    blank_image = np.zeros(size, dtype=np.uint8)
    return blank_image


def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None, height=None):
    # Define labels
    if not labels:
        labels = {0: u'wear helmet', 1: u'motorcycle', 2: u'not wear helmet', 3: u'plate'}
    # Define colors
    if not colors:
        colors = [(135, 171, 41), (67, 161, 255), (34, 34, 178), (186, 55, 2)]

    bb_motorcycle = []
    bb_no_helmet = []
    bb_plate = []

    # plot each boxes
    for box in boxes:
        # add score in label if score=True
        if score:
            label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%"
        else:
            label = labels[int(box[-1])]

        # filter every box under conf threshold if conf threshold setted
        if int(box[-1]) == 1:
            if height:
                convert_height = xyxy_to_xywh(box)
                if convert_height[-1] > height:
                    if conf:
                        if box[-2] > conf:
                            color = colors[int(box[-1])]
                            box_label(image, box, label, color)
                            bb_motorcycle.append(box)
                    else:
                        color = colors[int(box[-1])]
                        box_label(image, box, label, color)
                        bb_motorcycle.append(box)
            else:
                if conf:
                    if box[-2] > conf:
                        color = colors[int(box[-1])]
                        box_label(image, box, label, color)
                        bb_motorcycle.append(box)
                else:
                    color = colors[int(box[-1])]
                    box_label(image, box, label, color)
                    bb_motorcycle.append(box)
        elif int(box[-1]) == 0 or int(box[-1]) == 2:
            if conf:
                if box[-2] > conf:
                    if int(box[-1]) == 2:
                        bb_no_helmet.append(box)
                    color = colors[int(box[-1])]
                    box_label(image, box, label, color)
            else:
                if int(box[-1]) == 2:
                    bb_no_helmet.append(box)
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            if conf:
                if box[-2] > conf:
                    color = colors[int(box[-1])]
                    box_label(image, box, label, color)
                    bb_plate.append(box)
            else:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
                bb_plate.append(box)

    image_bb_motorcycle = [image, bb_motorcycle, bb_no_helmet, bb_plate]

    return image_bb_motorcycle
