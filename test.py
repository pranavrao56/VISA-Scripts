# Using Capsules dataset from the VisA dataset
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('capsules/image_anno.csv')
image_paths = data['image']

images = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    feature_vector = image.flatten()
    images.append(feature_vector)

X = np.array(images)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print('Accuracy Score: ',acc)
