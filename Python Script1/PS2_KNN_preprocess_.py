# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:40:54 2019
@author: liuyuanrong
"""
import numpy as np
import pandas as pd

# read csv as pandas dataframe
def read_data_pd(file_path):
    return pd.read_csv(file_path, header=None, sep=",")

# inner function for the precess_data, scaling the columns with numeric numbers:
#    precessed_dataframe is dataframe that have already precessed (eliminated null values with "?")
#    num_col_id is the id for columns of features that are numeric types, in the crx dataset, col id: 1,2,7,10,14 are continuous data
def real_valued_scaling(processed_dataframe, num_col_id = [1,2,7,10,13,14]):
    processed_dataframe[num_col_id] = processed_dataframe[num_col_id].apply(pd.to_numeric)
    temp_dataframe = processed_dataframe.copy()
    for col_id in num_col_id:
        temp = temp_dataframe[temp_dataframe.columns[col_id]]
        mu = np.mean(temp)
        sigma = np.std(temp)
        for row_id in range(len(temp)):
            temp_dataframe.iloc[row_id, col_id] = (processed_dataframe.iloc[row_id,col_id] - mu)/sigma
    return temp_dataframe

# precess raw data:
#    dataframe is a pandas dataframe
#    scaling is parameter to decide whether scaling the data (True) or not (False)
#    num_col_id is the id for columns of features that are numeric types, in the crx dataset, col id: 1,2,7,10,14 are continuous data
def process_data(dataframe, scaling = True, num_col_id = [1,2,7,10,13,14]):
    temp = dataframe.copy()
    data_labels = temp.iloc[:,-1]
    for col_id, col in enumerate(dataframe.columns):
        if col_id in num_col_id: # if the col is numeric
            col_temp = temp[col]
            col_mean = np.mean(col_temp[-col_temp.isin(["?"])].astype(float))
            for row_id in range(len(col_temp)):
                if col_temp.iloc[row_id] == "?":
                    temp.iloc[row_id,col_id] = col_mean
        else: # if the col is not numeric
            col_temp = temp[col]
            pos_val = pd.value_counts(col_temp[data_labels.isin(['+'])]).iloc[0] # most occurrenced value given "+"
            neg_val = pd.value_counts(col_temp[data_labels.isin(['-'])]).iloc[0] # most occurrenced value given "-"
            for row_id in range(len(col_temp)):
                if col_temp.iloc[row_id] == "?" and data_labels[row_id] == "+": # when the "?" row's label is "+"
                    temp.iloc[row_id, col_id] = pos_val
                elif col_temp.iloc[row_id] == "?" and data_labels[row_id] == "-": # when the "?" row's label is "-"
                    temp.iloc[row_id, col_id] = neg_val
    if scaling: # whether or not scaling the data after filled all "?"
        temp = real_valued_scaling(temp,num_col_id)
        return temp
    return temp

#data_path = "C:/Users/liuyu/Desktop/Spring2019/CS542/2019spring/problem _set/PS2/Homework_2/Homework_2/credit_2019/credit_2019/"
data_path = "./"
train_dataframe = read_data_pd(data_path+"crx.data.training")
test_dataframe = read_data_pd(data_path+"crx.data.testing")

num_columns = [1,2,7,10,13,14]
# precess train dataset
processed_train_dataframe = process_data(dataframe = train_dataframe, scaling = True, num_col_id = num_columns)
# process test dataset
processed_test_dataframe = process_data(dataframe = test_dataframe, scaling = True, num_col_id = num_columns)

processed_train_dataframe.to_csv(data_path+"processed_crx_data_training_final.csv",index=0)
processed_test_dataframe.to_csv(data_path+"processed_crx_data_testing_final.csv",index=0)