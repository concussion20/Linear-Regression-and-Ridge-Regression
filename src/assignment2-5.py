#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:


def z_score(dataset):
    means = dataset.mean(axis = 0, skipna = True)
    stds = dataset.std(axis = 0, skipna = True) 
    mean_list = list(means)[0:-1]
    std_list = list(stds)[0:-1]
    length = len(dataset.columns) - 1
    for i in range(length):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - mean_list[i]) / std_list[i]
    return list(means)[0:-1], list(stds)[0:-1], dataset
    
def z_score_with_paras(dataset, mean_list, std_list):
    length = len(dataset.columns) - 1
    for i in range(length):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - mean_list[i]) / std_list[i]
    return dataset

def add_ones(dataset):
    dataset.insert(0, 'ones', [1] * len(dataset))
    return dataset


# In[3]:


def polynomial_dataset(p, dataset):
    cols = dataset.columns
    col_len = len(cols) - 1
    for i in range(1, p):
        for j in range(col_len):
            dataset[cols[j] + str(i + 1)] = dataset[cols[j]] ** (i + 1)
    output = dataset[cols[-1]]
    dataset.drop(cols[-1], axis = 1, inplace = True)
    dataset[cols[-1]] = output
    return dataset


# In[4]:


def rmse(theta, dataset):
    predict_y = liner_model(dataset, theta)
    
    sse = 0
    for i in range(len(dataset)):
        sse += (predict_y[i] - dataset.iloc[i,-1]) ** 2
    res = np.sqrt(sse/len(dataset))
    return res

def liner_model(dataset, theta):
    res = []
    for i in range(len(dataset)):
        res.append(sum(theta * dataset.iloc[i,0:-1]))
    return res


# In[5]:


def normal_equation(dataset):
    theta = np.linalg.inv(np.dot(dataset.iloc[:,0:-1].T, dataset.iloc[:,0:-1])).dot(dataset.iloc[:,0:-1].T).dot(dataset.iloc[:,-1])
    return theta


# In[6]:


def k_folds_split(k, dataset):
    length = len(dataset)
    piece_len = int(length / k)
    mylist = list(range(length))
    random.shuffle(mylist)
    result = []
    for i in range(k):
        test_index = mylist[i*piece_len:(i+1)*piece_len]
        train_index = mylist[0:i*piece_len] + mylist[(i+1)*piece_len:]
        result.append((train_index, test_index))
    return result


# In[7]:


# Sinusoid dataset
feature_names = ['X', 'Y']
dataset = pd.read_csv('Assignment2/sinData_Train.csv', header = None, names = feature_names)
vaild_dataset = pd.read_csv('Assignment2/sinData_Validation.csv', header = None, names = feature_names)
prob_name = 'Sinusoid'

rmses = []
for i in range(15):
    copy_dataset = dataset.copy()
    copy_dataset = polynomial_dataset(i + 1, copy_dataset)
    copy_dataset = add_ones(copy_dataset)
    copy_vaild_dataset = vaild_dataset.copy()
    copy_vaild_dataset = polynomial_dataset(i + 1, copy_vaild_dataset)
    copy_vaild_dataset = add_ones(copy_vaild_dataset)
    theta = normal_equation(copy_dataset)
    rmse_valid = rmse(theta, copy_vaild_dataset)
    rmses.append(rmse_valid)
plt.plot(range(1, 16), rmses)
plt.xlabel('highest power')
plt.ylabel('rmse')
plt.show()


# In[11]:


# Yacht dataset
feature_names = ['1', '2', '3', '4', '5', '6', 'vals']
dataset = pd.read_csv('Assignment2/yachtData.csv', header = None, names = feature_names)
prob_name = 'Yacht'

mean_rmse_train_list = []
mean_rmse_test_list = []
for i in range(1, 8):
    train_rmses = []
    test_rmses = []
    for train_index, test_index in k_folds_split(10, dataset):
        train_data = dataset.iloc[train_index]
#         copy_train_data = train_data.copy()
        train_data = polynomial_dataset(i, train_data)
        means, stds, train_data = z_score(train_data)
        train_data = add_ones(train_data)
        theta = normal_equation(train_data)
        rmse_train = rmse(theta, train_data)
        
        test_data = dataset.iloc[test_index]
        test_data = polynomial_dataset(i, test_data)
        test_data = z_score_with_paras(test_data, means, stds)
        test_data = add_ones(test_data)
        rmse_test = rmse(theta, test_data)
        
        train_rmses.append(rmse_train)
        test_rmses.append(rmse_test)
#         print('train', rmse_train)
#         print('test', rmse_test)
#         print(i)
    # end for
    mean_train_rmse = np.mean(train_rmses)
    mean_test_rmse = np.mean(test_rmses)
    mean_rmse_train_list.append(mean_train_rmse)
    mean_rmse_test_list.append(mean_test_rmse)
plt.plot(range(1, 8), mean_rmse_train_list, range(1, 8), mean_rmse_test_list)
plt.xlabel('highest power')
plt.ylabel('rmse')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:




