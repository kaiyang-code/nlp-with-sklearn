import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

# data = pd.read_csv('/Users/ky/Documents/bill_authentication.csv')
#
#
# x = data.drop('Class', axis = 1)
# y = data['Class']
#
# print(x)
# print(y)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
# print('x_train: ', x_train)
# print('x_test: ', x_test)
# print('y_train: ', y_train)
# print('y_test: ', y_test)
#
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(x_train, y_train)
#
# y_pred = svclassifier.predict(x_test)
#
# print(y_pred)
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# the folowing is to use svm on the movie reviews

neg_path = '/Users/ky/Documents/review_polarity/txt_sentoken/neg'
pos_path = '/Users/ky/Documents/review_polarity/txt_sentoken/pos'
all_neg = os.listdir(neg_path)
all_pos = os.listdir(pos_path)
# the train_data and test_data consists of the actual classifications of the .txt files
train_class = []

# read spcific files from spcific folders
def read_in_files(files, path, start_pos, end_pos, class_id):
    data = []
    for i in range(start_pos, end_pos):
        file = open(path + '/' + files[i])
        temp = file.read().replace("\n", " ").lower()
        data.append(temp)
        file.close()
        if path == pos_path:
            class_id.append(1)
        else:
            class_id.append(0)
    return data

# the training data consisits of 5 positive reviews and five negative reviews
train_data = read_in_files(all_neg, neg_path, 0, 100, train_class)
train_data += read_in_files(all_pos, pos_path, 0, 100, train_class)

x_train, x_test, y_train, y_test = train_test_split(train_data, train_class, test_size = 0.30) # use random seeds to control randomness

vectorizer = CountVectorizer(stop_words = 'english')
vectorizer = vectorizer.fit(x_train + x_test)
# basically fit_transform is used on train_data and transfrom is used on test_data
train_features = vectorizer.transform(x_train)
test_features = vectorizer.transform(x_test)

# uses linear kernel to classify data
svclassifier = SVC(kernel='linear')
svclassifier.fit(train_features, y_train)

y_pred = svclassifier.predict(test_features)

print(y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# # uses polynomial kernel to classify data
# svclassifier = SVC(kernel='poly', degree = 8)
# svclassifier.fit(train_features, y_train)
#
# y_pred = svclassifier.predict(test_features)
#
# print(y_pred)
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))


