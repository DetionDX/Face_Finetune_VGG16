########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import vgg16_siamese
from sklearn.metrics import roc_curve, auc
from scipy import spatial
import matplotlib.pyplot as plt


def load_data(image_paths, pair_list):
    left = []
    right = []
    label=np.zeros(len(pair_list))
    c = 0
    for img in pair_list:
        item = img.split('\t')
        if len(item)==3:
            idx = format(int(item[1]),'04d')
            sub1 = image_paths+item[0]+'/'+ item[0]+'_'+ idx+ '.png'
            idx = format(int(item[2]), '04d')
            sub2 = image_paths+item[0]+'/'+ item[0]+'_'+ idx+ '.png'
            label[c] = 1
        elif len(item)==4:
            idx = format(int(item[1]),'04d')
            sub1 = image_paths+item[0]+'/'+ item[0]+'_'+ idx+ '.png'
            idx = format(int(item[3]), '04d')
            sub2 = image_paths+item[2]+'/'+ item[2]+'_'+ idx+ '.png'
            label[c] = -1
        c = c+1
        left.append(sub1)
        right.append(sub2)
    return left, right, label


def load_img(dice,left_path,right_path,label):
    left_img = []
    right_img = []
    out_label = []
    # print('loading data...')
    for i in dice:
        left_img.append(cv.imread(left_path[i],1))
        right_img.append(cv.imread(right_path[i], 1))
        out_label.append(label[i])
    left_img = np.stack(left_img)
    right_img = np.stack(right_img)

    return left_img, right_img, out_label

def load_test_img(test_images_l, test_images_r):
    left_img = cv.imread(test_images_l, 1)
    right_img = cv.imread(test_images_r, 1)

    return left_img, right_img

lr = 0.000000001
sess = tf.InteractiveSession()

weight = 'vgg16_weights.npz'

siamese = vgg16_siamese.siemese(weight)

train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(siamese.loss)

saver = tf.train.Saver()
tf.global_variables_initializer().run()


num_epoch = 10
# 2000 training samples
batch_size = 20
iteration_num  = 100

eval_iter = 1
train_iter = 1

image_path = '/home/labuser/Documents/lfw/lfw_mtcnnpy_224/'
train = 'pairsDevTrain.txt'
test = 'pairs.txt'

with open(train) as f:
    train_list = f.readlines()
train_list.pop(0)
with open(test) as f:
    test_list = f.readlines()
test_list.pop(0)

train_images_l,train_images_r, train_label = load_data(image_path, train_list)
test_images_l, test_images_r, test_label = load_data(image_path, test_list)

new = False
model_ckpt = 'model.ckpt'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = raw_input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

best_auc = 0
# start training
if new:
    for epoch in range(num_epoch):

        list = np.random.choice(len(train_images_l), len(train_images_l), replace=False)

        for step in range(iteration_num):
            dice = np.random.choice(list, batch_size, replace=False)
            batch_left, batch_right, batch_label = load_img(dice, train_images_l, train_images_r, train_label)
            _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                                siamese.x1: batch_left,
                                siamese.x2: batch_right,
                                siamese.y_: batch_label})

            if step % 10 == 0:
                print('this is epoch: [%d/%d] at step [%d/%d]' % (epoch, num_epoch, step, iteration_num))

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()

        if epoch % train_iter == 0 and epoch>0:
            print('training evaluating...')
            loss = 0
            score = np.zeros(len(train_label))
            for j in range(len(train_label)):
                left, right = load_test_img(train_images_l[j], train_images_r[j])

                loss = loss + siamese.loss.eval({siamese.x1: [left],siamese.x2:[right],siamese.y_: [train_label[j]]})

                feat_l = siamese.o1.eval({siamese.x1:[left]})
                feat_r = siamese.o2.eval({siamese.x2: [right]})
                dist = 1-spatial.distance.cosine(feat_l,feat_r)
                # print(dist)
                score[j] = dist

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fpr, tpr, _ = roc_curve(train_label, score)
            roc_auc = auc(fpr, tpr)
            print('this is epoch: [%d/%d] at step [%d/%d] with loss: %f, auc: %f' % (epoch, num_epoch, step, iteration_num, loss, roc_auc))

        if epoch % eval_iter == 0 and epoch > 0:
            print('test evaluating...')
            score = np.zeros(len(test_label))
            for j in range(len(test_label)):
                test_left, test_right = load_test_img(test_images_l[j], test_images_r[j])
                feat_lt = siamese.o1.eval({siamese.x1: [test_left]})
                feat_rt = siamese.o2.eval({siamese.x2: [test_right]})
                dist = 1-spatial.distance.cosine(feat_lt,feat_rt)
                score[j] = dist

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fpr, tpr, _ = roc_curve(test_label, score)
            roc_auc = auc(fpr, tpr)
            print('ROC curve auc for testing: %f' % roc_auc)
            # plt.show()
            # plt.plot(fpr, tpr, color='red', label='ROC curve (area = %f)' % roc_auc)
            # print('ROC auc %f' % roc_auc)
            # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.title('ROC for directly using vgg16 on lfw')
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            if roc_auc>=best_auc:
                best_auc=roc_auc
                print('best auc is: %f' % best_auc)
                saver.save(sess, 'model.ckpt')

else:
    saver.restore(sess, 'model.ckpt')
    print('test evaluating...')
    score = np.zeros(len(test_label))
    for j in range(len(test_label)):
        test_left, test_right = load_test_img(test_images_l[j], test_images_r[j])
        feat_lt = siamese.o1.eval({siamese.x1: [test_left]})
        feat_rt = siamese.o2.eval({siamese.x2: [test_right]})
        dist = 1 - spatial.distance.cosine(feat_lt, feat_rt)
        score[j] = dist

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(test_label, score)
    roc_auc = auc(fpr, tpr)
    print('ROC curve auc for testing: %f' % roc_auc)
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %f)' % roc_auc)
    print('ROC auc %f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC for directly using vgg16 on lfw')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

















