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


def liner_model(dataset, theta):
    res = []
    for i in range(len(dataset)):
        res.append(sum(theta * dataset.iloc[i,0:-1]))
    return res


# In[4]:


def update_theta(theta, alpha, dataset):
    predict_y = liner_model(dataset, theta)
    
    cols = len(dataset.columns) - 1
    rows = len(dataset)
    new_theta = [0] * cols
    for j in range(cols):
        for i in range(rows):
            new_theta[j] += (predict_y[i] - dataset.iloc[i,-1]) * dataset.iloc[i,j]
        new_theta[j] = theta[j] - alpha * new_theta[j] / rows
    return new_theta

def rmse(theta, dataset):
    predict_y = liner_model(dataset, theta)
    
    sse = 0
    for i in range(len(dataset)):
        sse += (predict_y[i] - dataset.iloc[i,-1]) ** 2
    res = np.sqrt(sse/len(dataset))
    return res

def bgd(dataset, tol, alpha, is_plot):
    cols = len(dataset.columns) - 1
    theta = [0] * cols
    max_iter = 1000
    new_loss = rmse(theta, dataset)
    loss = new_loss + 1000

    x_axis = []
    y_axis = []
    i = 0
    while np.abs(new_loss - loss) > tol and i < max_iter:
        x_axis.append(i + 1)
        y_axis.append(new_loss)
#         print(i)
#         print(new_loss - loss)
        theta = update_theta(theta, alpha, dataset)
        loss = new_loss
        new_loss = rmse(theta, dataset)
        i += 1
    # end while
    
    if is_plot:
        plt.plot(x_axis, y_axis)
        plt.xlabel('iteration')
        plt.ylabel('rmse')
        plt.show()
    
    return theta


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


def k_folds(dataset, tol, alpha):
    result_table = {}
    for i in range(1, 11):
        result_table[str(i)] = {}
        
    result_table2 = {}
    for i in range(1, 11):
        result_table2[str(i)] = {}
        
    result_table['mean RMSE'] = {}
    result_table['std RMSE'] = {}
    train_rmses = []
    test_rmses = []
    
    result_table2['mean RMSE'] = {}
    result_table2['std RMSE'] = {}
    train_rmses2 = []
    test_rmses2 = []
    i = 1
    for train_index, test_index in k_folds_split(10, dataset):
        train_data = dataset.iloc[train_index]
        means, stds, train_data = z_score(train_data)
        train_data = add_ones(train_data)
        theta = []
        if i <= 2:
            theta = bgd(train_data, tol, alpha, True)
        else:
            theta = bgd(train_data, tol, alpha, False)
#         print(theta)
        theta2 = normal_equation(train_data)
        rmse_train = rmse(theta, train_data)
        rmse_train2 = rmse(theta2, train_data)
        
        test_data = dataset.iloc[test_index]
        test_data = z_score_with_paras(test_data, means, stds)
        test_data = add_ones(test_data)
        rmse_test = rmse(theta, test_data)
        rmse_test2 = rmse(theta2, test_data)
        
        train_rmses.append(rmse_train)
        test_rmses.append(rmse_test)
        result_table[str(i)]['train'] = rmse_train
        result_table[str(i)]['test'] = rmse_test
        
        train_rmses2.append(rmse_train2)
        test_rmses2.append(rmse_test2)
        result_table2[str(i)]['train'] = rmse_train2
        result_table2[str(i)]['test'] = rmse_test2
        i += 1
    # end for
    mean_train_rmse = np.mean(train_rmses)
    std_train_rmse = np.std(train_rmses, ddof=1)
    mean_test_rmse = np.mean(test_rmses)
    std_test_rmse = np.std(test_rmses, ddof=1)
    result_table['mean RMSE']['train'] = mean_train_rmse
    result_table['std RMSE']['train'] = std_train_rmse
    result_table['mean RMSE']['test'] = mean_test_rmse
    result_table['std RMSE']['test'] = std_test_rmse
    
    mean_train_rmse2 = np.mean(train_rmses2)
    std_train_rmse2 = np.std(train_rmses2, ddof=1)
    mean_test_rmse2 = np.mean(test_rmses2)
    std_test_rmse2 = np.std(test_rmses2, ddof=1)
    result_table2['mean RMSE']['train'] = mean_train_rmse2
    result_table2['std RMSE']['train'] = std_train_rmse2
    result_table2['mean RMSE']['test'] = mean_test_rmse2
    result_table2['std RMSE']['test'] = std_test_rmse2
    columns = list(range(1, 11)) + ['mean RMSE', 'std RMSE']
    columns = [ str(x) for x in columns ]
    result_table_df = pd.DataFrame(result_table, index = ['train', 'test'], columns = columns )
    result_table_df2 = pd.DataFrame(result_table2, index = ['train', 'test'], columns = columns )
    print('use gradient descent')
    print(result_table_df)
    print('use normal equations')
    print(result_table_df2)


# In[8]:


# Housing dataset
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('Assignment2/housing.csv', header = None, names = feature_names)
prob_name = 'Housing'

tol = 0.5 * 1e-2
alpha = 0.4 * 1e-3
k_folds(dataset, tol, alpha)

from sklearn.linear_model import LinearRegression
means, stds, dataset = z_score(dataset)
dataset = add_ones(dataset)
lr = LinearRegression()
lr.fit(dataset.iloc[:,0:-1], dataset.iloc[:,-1])
print(lr.coef_)
print(lr.intercept_)
print(np.mean(dataset.iloc[:,-1]))
print(rmse(lr.coef_, dataset))


# In[9]:


# Yacht dataset
feature_names = ['1', '2', '3', '4', '5', '6', 'vals']
dataset = pd.read_csv('Assignment2/yachtData.csv', header = None, names = feature_names)
prob_name = 'Yacht'

tol = 0.1 * 1e-2
alpha = 0.1 * 1e-2
k_folds(dataset, tol, alpha)

from sklearn.linear_model import LinearRegression
means, stds, dataset = z_score(dataset)
dataset = add_ones(dataset)
lr = LinearRegression()
lr.fit(dataset.iloc[:,0:-1], dataset.iloc[:,-1])
print(lr.coef_)
print(lr.intercept_)
print(np.mean(dataset.iloc[:,-1]))
print(rmse(lr.coef_, dataset))


# In[ ]:




