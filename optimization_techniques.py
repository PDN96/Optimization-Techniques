import numpy as np
import pandas as pd
import random
import time
import math
import matplotlib.pyplot as plt

def read_all_file(file):
    data_file = file + '.csv'
    label_file = 'labels-' +file+ '.csv'
    data = pd.read_csv(data_file, header = None).values
    labels = pd.read_csv(label_file, header = None).values
    return data, labels

#Function to compute the accuracy of the models
def compute_acc(N_t, phi_test, test_label, SN, w):
    tp = tn = fp = fn = 0
    for i in range(N_t):
        mu_a = phi_test[i].dot(w)
        sigma_a_squared = phi_test[i].T.dot(SN.dot(phi_test[i]))
        kappa = (1 + np.pi * sigma_a_squared / 8) ** (-0.5)
        p = 1.0 / (1 + np.exp(- kappa * mu_a))
        if p >= 0.5:
            if test_label[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if test_label[i] == 0:
                tn += 1
            else:
                fn += 1
    acc = (tp + tn) / (tp + fp + tn + fn)
    return acc

#Function to split data but not randomly.
def split_data(all_data, all_labels):
    n = len(all_data)
    test_data = all_data[0:int(n / 3.0)]
    test_label = all_labels[0:int(n / 3.0)]
    train_data = all_data[int(n / 3.0):n]
    train_label = all_labels[int(n / 3.0):n]
    return test_data, test_label, train_data, train_label

#Function to implement gradient ascent.
def gradient_ascent(file):
    all_data, labels = read_all_file(file)
    test_data, test_label, train_data, t = split_data(all_data, labels)
    N = len(train_data)
    ones = np.array([[1]] * N)
    phi = np.concatenate((ones, train_data), axis=1)
    M = len(phi[0])
    eta = 10 **-3
    alpha = 0.1
    w = np.array([[0]] * M)
    update = 1
    n = 1
    time_spent = [0]
    I = np.eye(M)
    all_w = [w]
    start_time = time.time()

    while update > 10 ** -3 and n < 6000:
        w_old = w
        a = phi.dot(w_old)
        y = 1.0 / (1 + np.exp(-a))
        w_new = w_old - eta * (phi.T.dot(y - t) + alpha * w_old)
        update = np.linalg.norm(w_new - w_old) / np.linalg.norm(w_old)
        w = w_new
        n += 1
        if n % 50 == 0:
            all_w.append(w)
            time_spent.append(time.time() - start_time)

    N_t = len(test_data)
    ones = np.array([[1]] * N_t)
    test_data = np.concatenate((ones, test_data), axis=1)
    phi_test = test_data
    all_acc = []
    for w in all_w:
        a = phi.dot(w)
        y = 1.0 / (1 + np.exp(-a))
        SN_inv = alpha * I
        for n in range(N):
            SN_inv += y[n] * (1 - y[n]) * np.outer(phi[n], phi[n])
        SN = np.linalg.inv(SN_inv)
        acc = compute_acc(N_t, phi_test, test_label, SN, w)
        all_acc.append(acc)
    return time_spent, all_w, all_acc

#Function to implement Newton's method
def newton_method(file):
    all_data, labels = read_all_file(file)
    test_data, test_label, train_data, t = split_data(all_data, labels)
    N = len(train_data)
    ones = np.array([[1]] * N)
    phi = np.concatenate((ones, train_data), axis=1)
    M = len(phi[0])
    w = np.array([[0]]*M)
    update = 1
    n = 1
    alpha = 0.1
    time_spent = [0]
    all_w = []
    all_w.append(w)
    start_time = time.time()
    I = np.eye(M)

    while update > 10 ** -3 and n < 100:
        w_old = w
        a = phi.dot(w_old)
        y = 1.0 / (1 + np.exp(-a))
        r = y * (1 - y)
        R = np.diag(r.ravel())
        temp1 = phi.T.dot(y - t) + alpha * w_old
        temp2 = alpha * I + phi.T.dot(R.dot(phi))
        w_new = w_old - np.linalg.inv(temp2).dot(temp1)

        update = np.linalg.norm(w_new - w_old) / np.linalg.norm(w_old)
        time_spent.append(time.time() - start_time)
        w = w_new
        all_w.append(w)
        n += 1

    N_t = len(test_data)
    ones = np.array([[1]]*N_t)
    phi_test = np.concatenate((ones, test_data), axis=1)
    all_acc = []

    for w in all_w:
        a = phi.dot(w)
        y = 1.0 / (1 + np.exp(-a))
        SN_inv = alpha * I
        for i in range(N):
            SN_inv += y[i] * (1 - y[i]) * np.outer(phi[i], phi[i])
        SN = np.linalg.inv(SN_inv)
        acc = compute_acc(N_t, phi_test, test_label, SN, w)
        all_acc.append(acc)

    return time_spent, all_w, all_acc

datasets = ['A', 'usps']

#Plotting
for dataset in datasets:
    w_newton = list()
    w_gradient = list()
    time_newton = list()
    time_gradient = list()
    acc_newton = list()
    acc_gradient = list()
    for i in range(3):
        t_n, w_n, acc_n = newton_method(dataset)
        t_g, w_g, acc_g = gradient_ascent(dataset)
        time_newton.append(t_n)
        acc_newton.append(acc_n)
        time_gradient.append(t_g)
        acc_gradient.append(acc_g)
    time_newton = np.mean(np.array(time_newton), axis=0)
    time_gradient = np.mean(np.array(time_gradient), axis=0)
    acc_newton = np.mean(np.array(acc_newton), axis=0)
    acc_gradient = np.mean(np.array(acc_gradient), axis=0)
    print("Dataset :" + dataset)
    print("Accuracies and time for Newton method:")
    print(acc_newton)
    print(time_newton)
    print("Accuracies and time for Gradient Ascent:")
    print(acc_gradient)
    print(time_gradient)
    plt.gcf().clear()
    err_newton = [1 - x for x in acc_newton]
    err_gradient = [1 - x for x in acc_gradient]
    plt.plot(time_newton, err_newton, '-r', label = "Newton's Method")
    plt.plot(time_gradient, err_gradient, '-b', label = "Gradient Ascent")
    plt.xlabel("Run Time")
    plt.ylabel("Error Rate")
    plt.grid("on")
    plt.title("Dataset: " + dataset)
    plt.legend(loc="best")
    plt.savefig(dataset + '_p2.png')

#Function to implement stochastic gradient ascent
def stoc_ascent(file):
    all_data, labels = read_all_file(file)
    test_data, test_label, train_data, t = split_data(all_data, labels)
    N = len(train_data)
    ones = np.array([[1]] * N)
    phi = np.concatenate((ones, train_data), axis=1)
    M = len(phi[0])
    eta_o = 10 **-3
    alpha = 0.1
    w = np.array([0] * M)
    #update = 1
    I = np.eye(M)
    n = 1
    time_spent = [0]
    all_w = [w]
    start_time = time.time()

    while n < 6000:
        w_old = w
        stoc_i = random.randint(0,N-1)
        a = phi[stoc_i].dot(w_old)
        y = 1.0 / (1 + math.exp(-a))
        err = t[stoc_i] - y

        #dynamic eta formula
        eta = eta_o / (1 + n / N)
        #eta = eta_o * alpha ** (-n / N)

        #formula with constant eta and regularization
        #w_new = w_old + (N * eta_o * err * (phi[stoc_i]+ alpha * w_old.T)).T

        #formula with constant eta and no regularization
        #w_new = w_old + (N * eta_o * err * phi[stoc_i]).T

        #formula with dynamic eta and regularization
        #w_new = w_old + (N * eta * err * (phi[stoc_i]+ alpha * w_old.T)).T

        #formula with dynamic eta and no regularization
        w_new = w_old + (N * eta * err * phi[stoc_i]).T

        #update = np.linalg.norm(w_new - w_old) / np.linalg.norm(w_old)
        w = w_new
        n += 1
        if n % 50 == 0:
            time_spent.append(time.time() - start_time)
            all_w.append(w)
    N_t = len(test_data)
    ones = np.array([[1]] * N_t)
    phi_test = np.concatenate((ones, test_data), axis=1)
    all_acc = []
    for w in all_w:
        a = phi.dot(w)
        y = 1.0 / (1 + np.exp(-a))
        SN_inv = alpha * I
        for i in range(N):
            SN_inv += y[i] * (1 - y[i]) * np.outer(phi[i], phi[i])
        SN = np.linalg.inv(SN_inv)
        acc = compute_acc(N_t, phi_test, test_label, SN, w)
        all_acc.append(acc)
    return time_spent, all_w, all_acc

#Plotting
datasets = ['A', 'B', 'usps']
for dataset in datasets:
    time_s = list()
    time_gradient = list()
    acc_s = list()
    acc_gradient = list()
    for i in range(3):
        t_s, w_s, a_s = stoc_ascent(dataset)
        t_g, w_g, acc_g = gradient_ascent(dataset)
        time_s.append(t_s)
        acc_s.append(a_s)
        time_gradient.append(t_g)
        acc_gradient.append(acc_g)
    time_s = np.mean(np.array(time_s), axis=0)
    time_gradient = np.mean(np.array(time_gradient), axis=0)
    acc_s = np.mean(np.array(acc_s), axis=0)
    acc_gradient = np.mean(np.array(acc_gradient), axis=0)
    plt.gcf().clear()
    plt.plot(time_s, acc_s, '-r', label="Stochastic Gradient Ascent")
    plt.plot(time_gradient, acc_gradient, '-b', label="Gradient Ascent")
    plt.xlabel("Runtime")
    plt.ylabel("Accuracy")
    plt.grid("on")
    plt.title("Dataset: " +dataset)
    plt.legend(loc="best")
    plt.savefig(dataset + "_p3.png")
