import cv2
import keras
import numpy as np
import os
from keras.applications.inception_v3 import preprocess_input
from ana import generate_labelname_label_mapping

def generate_training_set(direct):
    image_name = os.listdir(direct)
    #mat = np.zeros((len(image_name), 256, 256, 3))
    mat = np.zeros((1,256, 256, 3))
    model = keras.models.load_model("inception-transferlearning_model6.h5")
    print("loaded model and weights...")
    image_name.sort()
    li = ['tea', 'fish', 'honey', 'juice', 'milk', 'nuts', 'sugar', 'jam', 'rice', 'coffee', 'oil', 'flour', 'corn', 'chocolate', 'water', 'cereal', 'pasta', 'chips', 'tomatosauce', 'vinegar', 'candy', 'beans', 'soda', 'cake', 'spices']
    i = 0
    print("image_id,label")
    for imagename in image_name:
        im = cv2.imread(direct+imagename)
        mat[0] = im
        mat[0] = preprocess_input(mat[0])
        prediction = model.predict(mat, batch_size = 1)
        #print(prediction)
        print(imagename[:-4] +',' +li[prediction.argmax()])
        i+=1


def test_on_train_set(direct):
    X = np.load('training_X')[:64]
    Y = np.load('training_Y')[:64]
    #mat = np.zeros((len(image_name), 256, 256, 3))
    mat = np.zeros((1,256, 256, 3))
    model = keras.models.load_model("initial_weights.h5")
    _ = {'tea': 0, 'fish': 1, 'honey': 2, 'juice': 3, 'milk': 4, 'nuts': 5, 'sugar': 6, 'jam': 7, 'rice': 8, 'coffee': 9, 'oil': 10, 'flour': 11, 'corn': 12, 'chocolate': 13, 'water': 14, 'cereal': 15, 'pasta': 16, 'chips': 17, 'tomatosauce': 18, 'vinegar': 19, 'candy': 20, 'beans': 21, 'soda': 22, 'cake': 23, 'spices': 24}

    li = ['tea', 'fish', 'honey', 'juice', 'milk', 'nuts', 'sugar', 'jam', 'rice', 'coffee', 'oil', 'flour', 'corn', 'chocolate', 'water', 'cereal', 'pasta', 'chips', 'tomatosauce', 'vinegar', 'candy', 'beans', 'soda', 'cake', 'spices']

    print(_, li)
    prediction = model.predict(X/255., batch_size = 16)
    correct = 0
    total = 0
    for x in range(len(Y)):
        print("original", Y[x], "predicted", prediction[x].argmax(), li[prediction[x].argmax()])
        if Y[x] == prediction[x].argmax():
            correct +=1
        total+=1
    print(correct, total, correct/total)

if __name__ == '__main__':
    generate_training_set('test_img/')
    #test_on_train_set('a')
    #generate_training_set('train_img/')
