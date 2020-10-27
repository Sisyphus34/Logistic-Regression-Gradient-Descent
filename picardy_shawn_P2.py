# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 22:12:59 2020

@author: sisyp
"""
import numpy as np
import matplotlib.pyplot as plt
import math



def process_data(filename):
    rows = 0;
    features = 0;
    
    data_file = open(filename, "r")
    
    first_line = data_file.readline().strip()
        
    data_params = first_line.split('\t')
    
    rows = int(data_params[0])
    features = int(data_params[1])
    prediction = 1
    
    x_data = np.zeros([rows, (features + prediction)])
    y_data = np.zeros([rows, 1])
    x_data_nosub0 = np.zeros([rows, features])

    last_col = features + prediction - 1
    
    for i in range(rows):
        row = data_file.readline().strip()
        x = row.split("\t")

        for j in range(features + prediction):
            if j == 0:
                x_data[i, j] = 1.0;
            else:
                x_data[i, j] = float(x[j - 1])
            if j == last_col:
                y_data[i, 0] = float(x[j])

        for j in range(features):
                x_data_nosub0[i, j] = float(x[j])

    data_file.close()
    
    return x_data, y_data
      

def gradient_descent(x, y):

    # number of set of data
    m = len(x)

    # number of features
    n = len(x[0])
    
    # approximate value for 'e'
    e = 2.71828
    J = 0
    # 1 x m vector of all ones (used to add up elements in a vector)
    Osubxm = np.ones((1, m))

    # (n + 1) x 1 vector of weights
    w = np.zeros((n, 1))

    # Alpha value 
    alpha = 0.095
    
    iterations = 20000
    for k in range(iterations):
        # exponent used in hypothesis function
        Xw = np.dot(x, w)
        # Hypothesis f(x). An m x 1 vector of hypothesis f(x) results
        H = 1 / (1 + (e**(-Xw)) )
    
        # Cost equation variables
        ln_H = H.copy()
        for p in range(len(ln_H)):
            ln_H[p][0] = math.log(H[p][0])
        
        one_minus_H = H.copy()
        for i in range(len(one_minus_H)):
            one_minus_H[i][0] = math.log(1 - H[i][0])

        # Cost is an m x 1 vector of cost incurred for each set of training data
        Cost =  -(np.multiply(y, ln_H)) - np.multiply((1 - y), one_minus_H)
    
        # 1x1 array containing one real number representation of Cost f(x) 
        J = (1/m) * np.dot(Osubxm, Cost)
      
        ##### Compute new weights #####
        w = w - (np.dot( ((alpha/m) * (H - y).T) , x) ).T
   
        if k == iterations -1:
            print("\n### Training Set Data ###")
            print("\nWeights = \n", w)
            print("\nFinal J = ", J)
            print("\nalpha = ", alpha)
            print("Iterations = ", iterations)

        if k % 100 == 0:
            plt.scatter(k, J, color= "green", marker = '.', label = "Type 1")

    plt.xlabel("Iterations")
    plt.ylabel("J-values")
    plt.title("Gradient Descent: J(w)")

    # Save plot
    plt.savefig('gradient_descent_P2.png.')

    plt.show()
    return H , J, w
    

def confusion_matrix(h_vals, y_vals):
    confusion_matrix  = [] 
    tp = 0 
    fp = 0
    tn = 0
    fn = 0

    for z in range(len(h_vals)):
        if h_vals[z] > 0.5:
            confusion_matrix.append(1)
        else:
            confusion_matrix.append(0)
    
    for el in range(len(confusion_matrix)):
        if confusion_matrix[el] == y_vals[el] and y_vals[el] == 0:
            tn += 1
        elif confusion_matrix[el] == y_vals[el] and y_vals[el] == 1:
            tp += 1
        elif confusion_matrix[el] != y_vals[el] and y_vals[el] == 0:
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  
    f1_score = 2 * (1 / ( (1 / precision) + (1 / recall) )) 

    print("\nFP = ", fp)
    print("TP =", tp)
    print("FN = ", fn)
    print("TN = ", tn)
    print("accuracy = ", accuracy)
    print("precision = ", precision)
    print("recall ", recall)
    print("F1 score = ", f1_score, "\n")



def project_two():
    e = 2.71828
    training_data = input("Please enter filename including extension for training data: \n")

    x, y  = process_data(training_data)

    hypothesis, final_J, weights = gradient_descent(x, y)

    confusion_matrix(hypothesis, y)

    test_data = input("Please enter filename including extension for test data: \n")

    x2, y2 = process_data(test_data)

    m = len(x2)
    n = len(x2[0])
    Osubxm = np.ones((1, m))

    Xw = np.dot(x2, weights)

    H = 1 / (1 + (e**(-Xw)) )

    # Cost equation variables
    ln_H = H.copy()
    for p in range(len(ln_H)):
        ln_H[p][0] = math.log(H[p][0])
    
    one_minus_H = H.copy()
    for i in range(len(one_minus_H)):
        one_minus_H[i][0] = math.log(1 - H[i][0])

    # Cost is an m x 1 vector of cost incurred for each set of training data
    Cost =  -(np.multiply(y2, ln_H)) - np.multiply((1 - y2), one_minus_H)

    # 1x1 array containing one real number representation of Cost f(x) 
    J = (1/m) * np.dot(Osubxm, Cost)
    print("\n### Test Set Data ###")
    confusion_matrix(H, y2)

    print("Testing Data J = ", J, "\n") 


project_two()




