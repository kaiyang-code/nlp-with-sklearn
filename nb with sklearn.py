import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split

neg_path = '/Users/ky/Documents/review_polarity/txt_sentoken/neg'
pos_path = '/Users/ky/Documents/review_polarity/txt_sentoken/pos'
all_neg = os.listdir(neg_path)
all_pos = os.listdir(pos_path)
# the train_data and test_data consists of the actual classifications of the .txt files
train_class = []
test_class = []

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

# vectorizer is used to convert a collection of text documents to a matrix of token counts
vectorizer = CountVectorizer(stop_words = 'english')
vectorizer = vectorizer.fit(train_data)
# basically fit_transform is used on train_data and transfrom is used on test_data
train_features = vectorizer.transform(x_train)
test_features = vectorizer.transform(x_test)


# uses the train_features and the train_class to train the model
nb = MultinomialNB()
nb.fit(train_features, y_train)
predictions = nb.predict(test_features)
print('predictions: ', predictions)

# confused?????????????
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))
