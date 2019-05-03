# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import confusion_matrix 
import numpy as np
from random import sample
import sys 
sys.path.append('C:/Users/liuyu/Desktop/Spring2019/CS542/2019spring/problem _set/PS4/0412') 
from new_SVM import loadMnist,SVM_classifier

# method to use the one vesus rest strategy to do multiple classification
def oneVSrest(input_data, input_label, test_data, test_label, sigma = .1, C = 15):
    class_type = np.unique(input_label)

    test_ret_mat = {} # store predicting values
    test_ret_label_mat = {} # store predicting labels
    test_ret_b_mat = {} # store b
    test_ret_w_mat = {}
    test_ret_true_label = {}
    
    for class_id in range(len(class_type)):
        current_class = class_type[class_id]
        #input_label_cp = label_transfer(input_label, current_class)
        input_label_cp = input_label.copy()
        positive_ids = np.where(input_label_cp == current_class)[0].tolist()
        negative_ids = np.where(input_label_cp != current_class)[0].tolist()
        np.put(input_label_cp, positive_ids, 1.0)
        np.put(input_label_cp, negative_ids, -1.0)    

        print(input_data.shape)    ###
        print(input_label_cp.shape)   ###
        
        print("================================================================class ", current_class, " versus rest===================================================================")
        classifier = SVM_classifier(kernal_options=["gaussian", sigma], opt_options=[15]) 
        classifier.fit(input_data, input_label_cp)   ###
        classifier.predict(test_data)   ###
       
        test_ret_b_mat[current_class] = classifier.b
        test_ret_w_mat[current_class] = classifier.w
        test_ret_mat[current_class] = classifier.ret_values.copy()
        test_ret_label_mat[current_class] = classifier.ret_labels.copy()
        test_ret_true_label[current_class] = test_label
        
        test_group = {"ret": test_ret_mat, "ret_label": test_ret_label_mat, "ret_b": test_ret_b_mat, "ret_w": test_ret_w_mat, "ret_true_label": test_ret_true_label, "sigma": classifier.sigma, "p": classifier.p, "C": classifier.C}
    return test_group

# method to use the one vesus one strategy to do multiple classification
def oneVSone(input_data, input_label, test_data, test_label, sigma, C):
    print("Sigma: ", sigma)
    print("C: ", C)
#    input_data = train_data_red
#    input_label = train_label_red
    class_type = np.unique(input_label)
    n_class, n_samples = len(class_type), input_data.shape[0]
    
    # generate a combination of one vs one classification group
    one_vs_one_groups = []
    for i in range(n_class):
        for j in range(i+1,n_class):
            one_vs_one_groups.append(str(int(class_type[i])) + "-" + str(int(class_type[j])))
    len(one_vs_one_groups)

    test_ret_mat = {} # store predicting values
    test_ret_label_mat = {} # store predicting labels
    test_ret_b_mat = {} # store b
    test_ret_w_mat = {}
    test_ret_true_label = {}
    
    ttt = 0
    for id in range(len(one_vs_one_groups)):
        ttt += 1
        current_class_pos = int(one_vs_one_groups[id][0])
        current_class_neg = int(one_vs_one_groups[id][-1])
        input_label_cp = input_label.copy()
        
        positive_ids = np.where(input_label_cp == current_class_pos)[0].tolist()
        negative_ids = np.where(input_label_cp == current_class_neg)[0].tolist()
        np.put(input_label_cp, positive_ids, 1.0)
        np.put(input_label_cp, negative_ids, -1.0)
        input_label_final = np.hstack((input_label_cp[positive_ids], input_label_cp[negative_ids]))
        input_data_final = np.vstack((train_data[positive_ids,:], train_data[negative_ids,:]))
        
        print("=================================================================", "classifier No.", ttt,": ", current_class_pos," vs ", current_class_neg,"=====================================================================")
        classifier = SVM_classifier(kernal_options=["gaussian", sigma], opt_options=[C]) 
        classifier.fit(input_data_final, input_label_final)
        classifier_name = one_vs_one_groups[id]
        
        print("testing set predicting: ", test_data.shape)
        print("testing label: ",test_label.shape)
        classifier.predict(test_data)
        
        test_ret_b_mat[classifier_name] = classifier.b
        test_ret_w_mat[classifier_name] = classifier.w
        test_ret_mat[classifier_name] = classifier.ret_values.copy()
        test_ret_label_mat[classifier_name] = classifier.ret_labels.copy()
        test_ret_true_label[classifier_name] = test_label
    
        test_group = {"ret": test_ret_mat, "ret_label": test_ret_label_mat, "ret_b": test_ret_b_mat, "ret_w": test_ret_w_mat, "ret_true_label": test_ret_true_label, "sigma": classifier.sigma, "p": classifier.p, "C": classifier.C}
    return test_group

# using the result of one vs one strategy, and perform an advance voting mechanism to classify the data, and summarize the result
def DAGSVM(test_data, test_label, test_results):
    bs = test_results['ret_b']
    ws = test_results['ret_w']
    y_predict_labels = []
    class_types = np.unique(test_label)
    
    for X in test_data:
        left, right = 0, len(class_types) - 1
    
        while left != right:
            paired_class = str(left) + "-" + str(right)
            temp_w, temp_b = ws[paired_class], bs[paired_class]
            y_predict = np.sign(np.dot(temp_w.T, X) + temp_b)
            if y_predict == -1:
                left += 1
            elif y_predict == 1:
                right -= 1
            else:
                raise ValueError("invalid y_predict;")
        y_predict_labels.append(left)
    cm = confusion_matrix(test_label, y_predict_labels)
    acc = get_acc(cm)
    return np.asarray(y_predict_labels), cm, acc

# randomly sample different indices
def sample_diff(ids, size):
    ret=[]
    for i in range(size):
        r = sample(ids, 1)[0]
        while r in ret:
            r = sample(ids,1)[0]
        ret.append(r)
    return ret

# get the maxnimum predict value's index for a data point in the one vs rest method
def grep_max_value_index(input_line, cl):
    max_id , max_val = -1, 0
    for id, val in enumerate(input_line):
        if val > max_val:
            max_id, max_val = id, val
    return(cl[max_id])

# transfer the 1, -1 binary classes to the 0,1,2,3,4,5,6,7,8,9, based on different classifiers. used in the one vs on method
def transfer_binary_to_label(input_line, class_types):
    labels = []
    for id, val in enumerate(input_line):
        if val == 1:
            label = int(class_types[id][0])
        elif val == -1:
            label = int(class_types[id][-1])
        else:
            raise ValueError("Invalid predict label;")
        labels.append(label)
    #print(labels)
    return labels

# compute accurancy form the confusion matrix
def get_acc(cm):
    n = cm.shape[0]
    correct_predict = 0
    for i in range(n):
        correct_predict += cm[i,i]
    return correct_predict / np.sum(cm)

# summarize one vs rest result
def ovr_test_results(test_results):
    test_y_predict_val = test_results['ret']
    test_y_predict = test_results['ret_label']
    test_y =  test_results['ret_true_label']
    
    test_y_m1 = pd.DataFrame(test_y_predict)
    test_y_m2 = pd.DataFrame(test_y)
    test_y_m3_val = pd.DataFrame(test_y_predict_val)
    
    class_types = list(range(10))
    predict_label = test_y_m3_val.apply(lambda x: grep_max_value_index(x, class_types) , 1)
    cm_test = confusion_matrix(test_y_m2[0].tolist(), predict_label)

    acc = get_acc(cm_test)
    return predict_label, cm_test, acc

# summarize one vs one result
def ovo_test_result_summary(test_results): 
    test_y_predict_val = test_results['ret']
    test_y_predict = test_results['ret_label']
    test_y =  test_results['ret_true_label']
    
    test_y_m1 = pd.DataFrame(test_y_predict)
    test_y_m2 = pd.DataFrame(test_y)
    test_y_m3_val = pd.DataFrame(test_y_predict_val)
    
    class_types = test_y_m1.columns.values
    
    predict_label_mat = test_y_m1.apply(lambda x: transfer_binary_to_label(x, class_types), 1)
    predict_label = predict_label_mat.apply(lambda x: pd.value_counts(x).index[0], 1)
    cm_test = confusion_matrix(test_y_m2['0-1'].tolist(), predict_label)
    acc = get_acc(cm_test)
    return np.asarray(predict_label), cm_test, acc

# start analysis
train_data, train_label, test_data, test_label = loadMnist("C:/Users/liuyu/Desktop/Spring2019/CS542/2019spring/problem _set/PS4/")
train_label , test_label = train_label.T.astype(float), test_label.T.astype(float)

train_label,test_label = train_label[0],test_label[0]

test_results = oneVSrest(train_data, train_label, test_data, test_label, sigma = 8, C = 10)
test_results = np.load("C:/Users/liuyu/Desktop/Spring2019/CS542/2019spring/problem _set/PS4/0412/result/1vr/ovr_test_results_3_18.npy").item()

y_predict_label_1vr, cm_1vr, acc_1vr = ovr_test_results(test_results)
print("one vs rest: ")
print(cm_1vr)
print(acc_1vr)

test_results = oneVSone(train_data, train_label, test_data, test_label, sigma = 5, C = 10)  
test_results = np.load("C:/Users/liuyu/Desktop/Spring2019/CS542/2019spring/problem _set/PS4/0412/result/one_vs_one_test_results_d5_20.npy").item()

y_predict_label_1v1, cm_1v1, acc_1v1 = ovo_test_result_summary(test_results)
y_predict_label_dag, cm_dag, acc_dag = DAGSVM(test_data, test_label, test_results)
print("one vs one: ")
print(cm_1v1)
print(acc_1v1)

print("dag: ")
print(cm_dag)
print(acc_dag)