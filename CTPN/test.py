# coding=utf-8
import os
import shutil
import sys
import time
import re
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import math
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
from math import *
import ocr
import preprocess_images
tf.app.flags.DEFINE_string('test_data_path', '.\\images', '')
tf.app.flags.DEFINE_string('output_path', '.\\result', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '.\checkpoints_mlt', '')
FLAGS = tf.app.flags.FLAGS


def straighten_image(img, degree, pt1, pt2, pt3, pt4):
    height, width, _ = img.shape
    fabs_sin, fabs_cos = fabs(sin(radians(degree))), fabs(cos(radians(degree)))
    height_new = int(width * fabs_sin + height * fabs_cos)
    width_new = int(height * fabs_sin + width * fabs_cos)
    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2
    img_rotation = cv2.warpAffine(img, mat_rotation, (width_new, height_new), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(mat_rotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(mat_rotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    img_out = img_rotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = img_out.shape[:2]
    return img_out

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_DUPLEX,
          font_scale=1,
          font_thickness=2,
          text_color=(0, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return img

def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def main(argv=None):
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)


            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))

                for i, box in enumerate(boxes):
                    box = box[:8].astype(np.int32).reshape(-1,2)
                    box = sort_poly(box.astype(np.int32))
                    pt1 = (box[0, 0], box[0, 1])
                    pt2 = (box[1, 0], box[1, 1])
                    pt3 = (box[2, 0], box[2, 1])
                    pt4 = (box[3, 0], box[3, 1])
                    crop_img = straighten_image(img, math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)
                    preprocessed_image = preprocess_images.preprocess_image(crop_img)
                    detect_text = ocr.tesseract(preprocessed_image)
                    detect_text = re.sub("[^a-zA-Z0-9]","",detect_text)
                    # num_digits = sum(list(map(lambda x:1 if x.isdigit() else 0,set(detect_text))))
                    # if num_digits<3:
                    #     continue
                    detect_text = detect_text.upper()
                    print('Detected result: {}'.format(detect_text))
                    bottomLeftCornerOfText = (box[0, 0], box[0, 1]-30)
                    cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                    img1 = draw_text(img, detect_text, bottomLeftCornerOfText)
                    
                img_path = os.path.join(FLAGS.output_path, os.path.basename(im_fn))
                cv2.imwrite(img_path, img1)

                with open(os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                          "w") as f:
                    for i, box in enumerate(boxes):
                        line = ",".join(str(box[k]) for k in range(8))
                        line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)


if __name__ == '__main__':
    tf.app.run()
