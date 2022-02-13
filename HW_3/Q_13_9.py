from numpy.core.fromnumeric import argmax
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from skimage import exposure
from sklearn.datasets import fetch_openml
from optimizers_only import gradient_descent_nn
from optimizers_only import gradient_descent
from sklearn.model_selection import train_test_split
np.random.seed(27)
# load in dataset
datapath = 'Data/'
csvname = datapath + 'noisy_sin_sample.txt'
data = np.loadtxt(csvname, delimiter = ',')
x = data[:-1,:]
y = data[-1:,:]
x_orig = x
y_orig = y
print(np.shape(x))
print(np.shape(y))


# Normalise inputs
for i in range(np.size(x,0)):    
    row_mean = np.mean(x[i,:])
    row_std = np.std(x[i,:])
    x[i,:] = x[i,:]-row_mean/row_std 

x_train, x_valid, y_train, y_valid = train_test_split(x.T, y.T, test_size=0.33)
#print(np.shape(x_train))
#print(np.shape(x_valid))
#breakpoint()
x = x_train.T
y = y_train.T
x_valid = x_valid.T
y_valid = y_valid.T
# neural network feature transformation
def feature_transforms(a, w):
    # loop through each layer
    for W in w:
        # compute inner -product with current layer weights
        a = W[0] + np. dot(a.T , W[1:])
        # pass through activation
        a = np.tanh(a). T
    return a

# neural network model
#def model(x, theta):
def model(theta,x):
    # compute feature transformation
    f = feature_transforms(x, theta[0])
    # compute final linear combination
    a = theta[1][0] + np. dot(f.T, theta [1][1:])
    return a.T

# create initial weights for a neural network model
def network_initializer(layer_sizes, scale):
    # container for all tunable weights
    weights = []
    # create appropriately -sized initial
    # weight matrix for each layer of network
    for k in range(len(layer_sizes) -1):
        # get layer sizes for current weight matrix
        U_k = layer_sizes[k]
        U_k_plus_1 = layer_sizes[k +1]
        # make weight matrix
        weight = scale* np. random. randn(U_k+ 1, U_k_plus_1)
        weights. append(weight)

    # repackage weights so that theta_init[0] contains all
    # weight matrices internal to the network, and theta_init[1]
    # contains final linear combination weights
    theta_init = [weights[:-1], weights[-1]]

    return theta_init

   
def rmse_func(w) :
    cost = 0
    y_pred = model(w,x)
    #y_pred = y_pred._value
    for p in range(y.size):
       # get pth prediction/ output pair
       y_p = y[:,p]
       y_model = y_pred[:,p]
       ## add to current cost
       cost += (y_p - y_model)**2
    return cost/ float(np.size(y))


# Initialise as per Example 13.4 from text
layer_sizes = [np.shape(x)[0],10,10,10,1]
w_init = network_initializer(layer_sizes,1.0)
max_its = 1000
alpha_choice = 0.05
weight_history_1k,cost_history_1k = gradient_descent_nn(rmse_func,alpha_choice,max_its,w_init)

max_its = 2000
alpha_choice = 0.05
weight_history_2k,cost_history_2k = gradient_descent_nn(rmse_func,alpha_choice,max_its,w_init)

max_its = 3000
alpha_choice = 0.05
weight_history_3k,cost_history_3k = gradient_descent_nn(rmse_func,alpha_choice,max_its,w_init)

max_its = 4000
alpha_choice = 0.05
weight_history_4k,cost_history_4k = gradient_descent_nn(rmse_func,alpha_choice,max_its,w_init)

max_its = 5000
alpha_choice = 0.05
weight_history_5k,cost_history_5k = gradient_descent_nn(rmse_func,alpha_choice,max_its,w_init)

max_its = 6000
alpha_choice = 0.05
weight_history_6k,cost_history_6k = gradient_descent_nn(rmse_func,alpha_choice,max_its,w_init)

max_its = 7000
alpha_choice = 0.05
weight_history_7k,cost_history_7k = gradient_descent_nn(rmse_func,alpha_choice,max_its,w_init)



#plt.figure(4)
#plt.plot(cost_history)
#plt.title('Cost history vs Iteration')
#plt.xlabel('Iteration')
#plt.ylabel('Cost')
#plt.figure()
##plt.show()


#y_p = model(weight_history[-1],x_valid)
y_p_1k = model(weight_history_1k[-1],x_valid)
y_p_2k = model(weight_history_2k[-1],x_valid)
y_p_3k = model(weight_history_3k[-1],x_valid)
y_p_4k = model(weight_history_4k[-1],x_valid)
y_p_5k = model(weight_history_5k[-1],x_valid)
y_p_6k = model(weight_history_6k[-1],x_valid)
y_p_7k = model(weight_history_7k[-1],x_valid)
print('Accuracy calc')
print(np.shape(y_p_1k))
print(np.shape(y_valid))
breakpoint()

def error_calc(y_pred_tmp) :
    error = (y_pred_tmp - y_valid)**2
    err_acc = np.sum(error)
    return err_acc

err_1k = error_calc(y_p_1k)
err_2k = error_calc(y_p_2k)
err_3k = error_calc(y_p_3k)
err_4k = error_calc(y_p_4k)
err_5k = error_calc(y_p_5k)
err_6k = error_calc(y_p_6k)
err_7k = error_calc(y_p_7k)

err_all_iter = [err_1k, err_2k, err_3k, err_4k, err_5k, err_6k, err_7k]
iter_plt = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
plt.figure(4)
plt.plot(iter_plt, err_all_iter,marker=".",)
plt.title('Validation error vs Iteration')
plt.xlabel('Iterations')
plt.ylabel('Validation Error')
#plt.show()
print(err_all_iter)
plt.figure(1)
plt.scatter(x_valid[0,:],y_p_3k[0,:], s=50, c='b', marker="s", label='3k iter model fit')
plt.scatter(x_valid[0,:],y_valid[0,:], s=50, c='r', marker="o", label='original')
plt.legend(loc='upper left')
plt.title ('Validation Dataset for 3000 iteration')

plt.figure(2)
y_p = model(weight_history_3k[-1],x)
print(np.shape(y_p))
plt.scatter(x[0,:],y_p[0,:], s=50, c='b', marker="s", label='3k iter model fit')
plt.scatter(x[0,:],y[0,:], s=50, c='r', marker="o", label='original')
plt.legend(loc='upper left')
plt.title ('Training Dataset for 3000 iteration')

plt.figure(3)
y_p = model(weight_history_3k[-1],x_orig)
print(np.shape(y_p))
plt.scatter(x_orig[0,:],y_p[0,:], s=50, c='b', marker="s", label='3k iter model fit')
plt.scatter(x_orig[0,:],y_orig[0,:], s=50, c='r', marker="o", label='original')
plt.legend(loc='upper left')
plt.title ('Total Dataset for 3000 iteration')

plt.show()