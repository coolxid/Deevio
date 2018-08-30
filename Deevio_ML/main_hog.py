import os
import sys
import cv2
import json
import imutils
import pickle
import urllib
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
def read_train(dataset_path):
    # --- Getting the training labels ---
    train_labels = os.listdir(dataset_path)
    print(train_labels)

    # --- Empty lists to hold images and labels ---
    labels = []
    hog_features = []

    for i in range(len(train_labels)):
        '''
            Join the training data path and each species training folder
            Dir = os.path.join(dataset_path, current_label)
            Get the current training label
        '''
        current_label = train_labels[i]
        train_images = [f for f in os.listdir(os.path.join(dataset_path, current_label))
                        if f.endswith('.jpeg')]
        # --- Num of images per class ---
        images_per_class = len(train_images)

        # --- Loop over the images in each sub-folder ---
        for j in range(images_per_class):
            # --- setting the image path ---
            dir = os.path.join(dataset_path, current_label)
            file = dir + '/' + train_images[j]

            # -- Read the image ---
            #im = Image.open(file)
            #arr = np.fromiter(iter(image.getdata()), np.uint8)
            #arr.resize(image.height, image.width)

            image = cv2.imread(file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = imutils.auto_canny(gray)

            '''
                Find contours in the edge map, keeping only the largest one which 
                is presmumed to be the car logo
            '''
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            c = max(cnts, key=cv2.contourArea)

            # --- Extract the nail and resize it to a canonical width and height ---
            (x, y, w, h) = cv2.boundingRect(c)
            nail = gray[y:y + h, x:x + w]
            nail = cv2.resize(nail, (200, 100))

            fd = hog(nail, orientations=9, pixels_per_cell=(10, 10),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
            hog_features.append(fd)
            labels.append(current_label)

        print("Folder processed: {}".format(current_label))
    print("Completed loading dataset...")
    return np.asarray(hog_features), np.asarray(labels)

def read_test(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    '''
        Find contours in the edge map, keeping only the largest one which 
        is presmumed to be the car logo
    '''
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # --- Extract the nail and resize it to a canonical width and height ---
    (x, y, w, h) = cv2.boundingRect(c)
    nail = gray[y:y + h, x:x + w]
    nail = cv2.resize(nail, (200, 100))

    fd = hog(nail, orientations=9, pixels_per_cell=(10, 10),
             cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    return fd.reshape(1, -1)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        dataset_Path = sys.argv[1]
        mode = sys.argv[2]

    else:
        #dataset_Path = 'Data1/bad/1522141919_bad.jpeg'
        dataset_Path = 'Data1/good/1522072948_good.jpeg'
        mode = 'test'
        # dataset_Path = 'Data1/'
        # mode = 'train'

    if mode == 'train':
        X, Y = read_train(dataset_Path)

        # --- Splitting data into training and test set ---
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        print("Training dataset dimensions {}, Training Label dimensions {} "
              .format(X_train.shape, y_train.shape))
        print("Test dataset dimensions {}, Test Label dimensions {} "
              .format(X_test.shape, y_test.shape))

        rf_model = RandomForestClassifier(n_estimators=109, random_state=0)
        rf_model.fit(X_train, y_train)
        predicted = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, predicted)
        print('Random Forest Mean accuracy score: {:.3}'.format(accuracy))
        print('\n')
        print(classification_report(y_test, predicted))
        pickle.dump(rf_model, open('model_ml.pkl', 'wb'))

    elif mode == 'test':
        X = read_test(dataset_Path)
        model = pickle.load(open('model_HOG.pkl', 'rb'))

        y_pred = model.predict(X)
        print(y_pred)

        with open('data.json', 'w') as outfile:
            json.dump(y_pred[0], outfile)

    else:
        print('Incorrect Mode....')
