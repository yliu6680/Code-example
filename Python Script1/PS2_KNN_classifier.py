# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:37:27 2019
@author: liuyuanrong
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# transfer the charcter to int:
#    processed_data is pandas dataframe, 
#    the num_col_id is the id for columns of features that are numeric types, in the crx dataset, col id: 1,2,7,10,14 are continuous data
def relevel_non_num_col(processed_data, num_col_id = [1,2,7,10,14]):
    input_data = processed_data.copy()
    for col_id, col in enumerate(processed_data.columns):
        if col_id not in num_col_id:
            keys = processed_data[col].unique().tolist()
            values = list(range(len(keys)))
            dictionary = dict(zip(keys,values))
            input_data[col] =  processed_data[col].map(dictionary)
    return input_data.values

# divide the data into features and label 
def divide_data(processed_data):
    return processed_data.iloc[:,:(len(processed_data.columns) - 1)], processed_data.iloc[:,-1]

# compute the euclidean distances of data ooint 1 and data point 2:
#    data_point1, data_point2 are 2 data points from the datasets that have same data structure
#    the num_col_id is the id for columns of features that are numeric types, in the crx dataset, col id: 1,2,7,10,14 are continuous data
def KNN_DL2_distance(data_point1,data_point2,num_col_id = [1,2,7,10,14]):
    distance = 0
    for i in range(len(data_point1)):
        if i in num_col_id:
            distance += np.square(data_point1[i] - data_point2[i])
        else:
            if data_point1[i] == data_point2[i]:
                distance += 0
            else:
                distance += .1 # the distances of different categorical values are set to 0.1
    distance = np.sqrt(distance)
    return distance

# KNN classifier:
#    the upper characters are training data, X is the training features, Y is the training label
#    the lower characters are testing data, x is the testing features.
#    the k is the number of K in KNN algorithm
#    the num_col_id is the id for columns of features that are numeric types, in the crx dataset, col id: 1,2,7,10,14 are continuous data
def KNN_predictor(X, Y, k, x, num_col_id = [1,2,7,10,14]):
    y_predict = []
    for test_dp_id in range(len(x)):
        distances = []
        count_labels = {}
        for train_dp_id in range(len(X)):
            distances.append(KNN_DL2_distance(x[test_dp_id], X[train_dp_id],num_col_id = num_col_id))
        sorted_indices = np.array(distances).argsort()
        indicies = sorted_indices[:k]
        for id in indicies:
            Y_label = Y[id]
            count_labels[Y_label] = count_labels.get(Y_label, 0) + 1
        y_predict.append(sorted(count_labels.items(),reverse = True,key = lambda d:d[1])[0][0])
    return y_predict

# compute the accurency for the KNN algorithm:
#    y is the true labels for the dataset, y_predict are the predicted labels from the KNN_predictor function
def KNN_evaluate(y, y_predict):
    count = 0
    for id, i in enumerate(y_predict):
        if y_predict[id] == y[id]:
            count += 1
    result = count/len(y_predict)
    return result

# output the result as described in the homework pdf file:
#    y_predict is the predicted label, x is the testing feature
def KNN_output(y_predict, x):
    x = pd.DataFrame(x)
    x["y_predict"] = y_predict
    return x

data_path = "C:/Users/liuyu/Desktop/Spring2019/CS542/2019spring/problem _set/PS2/Homework_2/Homework_2/credit_2019/credit_2019/"
#data_path = "./"
train_data = pd.read_csv(data_path+"processed_crx_data_training_final.csv")
test_data = pd.read_csv(data_path+"processed_crx_data_testing_final.csv")

# preprocess and devide the data
X, Y = divide_data(train_data)
x, y = divide_data(test_data)
num_colomns = [1,2,7,10,13,14]
X = relevel_non_num_col(X,num_colomns)
x = relevel_non_num_col(x,num_colomns)

# predict labels, get and save the result, and visualize it
ks = list(range(1,14))
test_result, train_result = [], []
for k in ks:
    print("running ",k)
    y_predict = KNN_predictor(X,Y,k,x,num_colomns)
    Y_predict = KNN_predictor(X,Y,k,X,num_colomns)
    test_result.append(KNN_evaluate(y,y_predict))
    train_result.append(KNN_evaluate(Y,Y_predict))
final_result = pd.DataFrame({"K number": ks, "Test dataset":test_result, "Train dataset":train_result})
final_result.to_csv(data_path + "KNN_final_result.csv")

x = ks
y_test = test_result
y_train = train_result
lines = plt.plot(x, y_test, x, y_train)
plt.legend(('Test dataset', 'Train dataset'),loc='upper right')
plt.title('Accurancy for testing and training dataset')
plt.xlabel("Number of k")
plt.ylabel("Accurancy")
plt.savefig(data_path+"KNN_final_result.jpg")
plt.show()

y_predict = KNN_predictor(X,Y,9,x,num_colomns)
KNN_output(y_predict, x).to_csv(data_path+"testing_data_with_y_predict.csv")