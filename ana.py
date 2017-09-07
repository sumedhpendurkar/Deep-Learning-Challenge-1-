import cv2
import numpy as np
import os
import pickle

def generate_labelname_label_mapping():
    a = open('train.csv', 'r')
    l = []
    di = {}
    labels = set() 
    for x in a:
        image_name, label =  x.split(',')
        #print image_name, label
        labels.add(label[:-1])
    labels.remove('label')
    i = 0
    for j in labels:
        di[j] = i
        i += 1
        l.append(j)
    return di, l



def generate_training_set(direct):
    image_name = os.listdir(direct)
    mat = np.zeros((len(image_name), 256, 256, 3))
    labels = np.zeros((len(image_name)))
    a = open('train.csv', 'r')
    a = a.readlines()
    a = a[1:]
    a.sort()
    image_name.sort()
    di, _ = generate_labelname_label_mapping()
    i = 0
    for imagename in image_name:
        im = cv2.imread(direct+imagename)
        mat[i] = im
        labels[i] = di[a[i].split(',')[1][:-1]]
        i+=1
    return mat, labels

def wrapper():
    mat, label = generate_training_set('train_img/')
    pickle.dump({'x':mat, 'label': label}, open("dataset","wb"))

if __name__ == "__main__":
    wrapper()
