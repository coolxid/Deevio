import cv2
import json
import urllib
import pickle
import imutils
from PIL import Image
import numpy as np
from skimage.feature import hog


def read_test(image_path):

    resp = urllib.request.urlopen(image_path)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    '''
        Find contours in the edge map, keeping only the largest one which 
        is presmumed to be the nail.
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

def do_pred(url):

    #dataset_Path = 'Data1/bad/1522141919_bad.jpeg'
    #dataset_Path = 'Data1/good/1522072948_good.jpeg'
    #mode = 'test'

    X = read_test(url)
    model = pickle.load(open('model_HOG.pkl', 'rb'))

    y_pred = model.predict(X)
    return y_pred[0]

if __name__ == '__main__':
    do_pred(url)
    #http://127.0.0.1:5000/predict?image=http://127.0.0.1/Data1/bad/1522141919_bad.jpeg


