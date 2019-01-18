import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
#See the data from the respitory about BM25 with C++
#remodelling the SVM format of training dataset into array format
def get_data1():
    data = load_svmlight_file("train.svm")
    return data[0], data[1]

X1, y1 = get_data1()
X1=X1[:,0:50812]

#remodelling the SVM format of testing dataset into array format
def get_data2():
    data = load_svmlight_file("test.svm")
    return data[0], data[1]

X2, y2 = get_data2()

#training the random forest model for the classifier

from sklearn.ensemble import RandomForestClassifier
import time
start_time = time.time()

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_jobs=2, random_state=200)
# Train the model on training data
rf.fit(X1, y1);
print("--- %s seconds ---" % (time.time() - start_time))

#show the prediction
predictions = rf.predict_proba(X2)

error=0
for i in range(len(predictions)):
    if predictions[i] != y2[i]:
        error=error+1
rate= 100* (error / len(predictions))

# Print out the mean absolute error (mae)
print('Mean Absolute Error for Random forest classifier:', rate, 'percent')
#combine it with the prediction of SVM
error=0
with open('train.svm.scale.predict') as fs:
    SVM_predict=fs.readlines()
print(len(SVM_predict))
print(len(predictions))
for i in range(len(predictions)):
    if predictions[i] != y2[i] or predictions[i] != SVM_predict[i]:
        error=error+1
rate= 100* (error / len(predictions))
print('Mean Absolute Error:', 100-rate, 'percent')
