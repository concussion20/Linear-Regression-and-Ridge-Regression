#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import random


# In[3]:


def center_data(dataset):
    means = dataset.mean(axis = 0, skipna = True)
    mean_list = list(means)
    length = len(dataset.columns)
    for i in range(length):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - mean_list[i])
    return list(means), dataset
    
def center_data_with_paras(dataset, mean_list):
    length = len(dataset.columns)
    for i in range(length):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - mean_list[i])
    return dataset

# def add_ones(dataset):
#     dataset.insert(0, 'ones', [1] * len(dataset))
#     return dataset


# In[4]:


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


# In[5]:


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


# In[6]:


def ridge_regression(dataset, lam):
    theta = np.linalg.inv(np.dot(dataset.iloc[:,0:-1].T, dataset.iloc[:,0:-1]) + 
                          lam * np.identity(dataset.iloc[:,0:-1].shape[1])).dot(dataset.iloc[:,0:-1].T).dot(dataset.iloc[:,-1])
    return theta


# In[7]:


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


# In[9]:


# Sinusoid dataset
feature_names = ['X', 'Y']
dataset = pd.read_csv('Assignment2/sinData_Train.csv', header = None, names = feature_names)
vaild_dataset = pd.read_csv('Assignment2/sinData_Validation.csv', header = None, names = feature_names)
prob_name = 'Sinusoid'

lambdas = np.arange(0, 10, 0.2)
rmses = []
mean_rmse_train_list = []
mean_rmse_test_list = []
mean_rmse_train_list2 = []
mean_rmse_test_list2 = []
for i in range(len(lambdas)):
    lam = lambdas[i]
    
    train_rmses = []
    test_rmses = []
    train_rmses2 = []
    test_rmses2 = []
    for train_index, test_index in k_folds_split(10, dataset):
        train_data = dataset.iloc[train_index]
        
        # maxp = 5
        copy_train_data = train_data.copy()
        copy_train_data = polynomial_dataset(5, copy_train_data)
        means, copy_train_data = center_data(copy_train_data)
#         print(copy_train_data)
        # train_data = add_ones(train_data)
        theta = ridge_regression(copy_train_data, lam)
#         theta.insert(0, means[-1])
        rmse_train = rmse(theta, copy_train_data)
        
        test_data = dataset.iloc[test_index]
        test_data = polynomial_dataset(5, test_data)
        test_data = center_data_with_paras(test_data, means)
        # test_data = add_ones(test_data)
        rmse_test = rmse(theta, test_data)
        
        train_rmses.append(rmse_train)
        test_rmses.append(rmse_test)
        
#         print('lambda is', lam)
#         print('my train rmse', rmse_train)
        from sklearn.linear_model import Ridge
        ridge_reg = Ridge(alpha=lam, solver="cholesky")
        ridge_reg.fit(copy_train_data.iloc[:,0:-1], copy_train_data.iloc[:,-1])
#         print('sklearn train rmse', rmse(ridge_reg.coef_, copy_train_data), ridge_reg.intercept_, ridge_reg.coef_)
        
        # maxp = 9
        copy_train_data = train_data.copy()
        copy_train_data = polynomial_dataset(9, copy_train_data)
        means, copy_train_data = center_data(copy_train_data)
        # train_data = add_ones(train_data)
        theta = ridge_regression(copy_train_data, lam)
        rmse_train2 = rmse(theta, copy_train_data)
        
        test_data = dataset.iloc[test_index]
        test_data = polynomial_dataset(9, test_data)
        test_data = center_data_with_paras(test_data, means)
        # test_data = add_ones(test_data)
        rmse_test2 = rmse(theta, test_data)
        
        train_rmses2.append(rmse_train2)
        test_rmses2.append(rmse_test2)
#         print('train', rmse_train)
#         print('test', rmse_test)
#         print(i)
    # end for
    mean_train_rmse = np.mean(train_rmses)
    mean_test_rmse = np.mean(test_rmses)
    mean_rmse_train_list.append(mean_train_rmse)
    mean_rmse_test_list.append(mean_test_rmse)
    
    mean_train_rmse2 = np.mean(train_rmses2)
    mean_test_rmse2 = np.mean(test_rmses2)
    mean_rmse_train_list2.append(mean_train_rmse2)
    mean_rmse_test_list2.append(mean_test_rmse2)
# end outer for
plt.plot(lambdas, mean_rmse_train_list)
plt.xlabel('lambda')
plt.ylabel('train rmse')
plt.title('power 5 train rmse')
plt.show()

plt.plot(lambdas, mean_rmse_test_list)
plt.xlabel('lambda')
plt.ylabel('test rmse')
plt.title('power 5 test rmse')
plt.show()

plt.plot(lambdas, mean_rmse_train_list2)
plt.xlabel('lambda')
plt.ylabel('train rmse')
plt.title('power 9 train rmse')
plt.show()

plt.plot(lambdas, mean_rmse_test_list2)
plt.xlabel('lambda')
plt.ylabel('test rmse')
plt.title('power 9 test rmse')
plt.show()


# In[ ]:




