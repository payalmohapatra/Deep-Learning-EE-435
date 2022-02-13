from numpy.core.fromnumeric import argmax
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.datasets import fetch_openml
from pandas import DataFrame
from optimizers_only import gradient_descent
from optimizers_batch import gradient_descent_batch

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# convert string labels to integers
y = np.array([int(v) for v in y])[:,np.newaxis]
x = DataFrame(x).to_numpy()
x = x.T
y = y.astype(int)
for i in range(np.shape(x)[0]):
    x[i,:] = (x[i,:] - np.mean(x[i,:]))/(np.std(x[i,:]))

x = DataFrame(x)
x = x.dropna()
x = DataFrame(x).to_numpy()

np.random.seed(0)

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

N = np.shape(x)[0] 
# print('N is :', N)
U_1 = 10
U_L = 10
C = 10  ## if C is > 0 it is +1 class and if it is < 0 then -1 class
layer_sizes = [N, U_1, U_1, U_1, U_1, C]
scale = 1

theta_init = network_initializer(layer_sizes, scale)

def multisoftmax(a, y):
    
    index = np.array([range(np.size(y))]).T
    exp_a = np.reshape(np.log(np.sum(np.exp(a), axis=1)), (np.size(y),1)) - a[index,y.T]
    cost = np.sum(exp_a)    
    
    return cost/float(np.size(y))

# # neural network feature transformation
def feature_transforms(a, w):
    # loop through each layer
    for W in w:
    # compute inner -product with current layer weights
        a = W[0] + np. dot(a.T , W[1:])        
        # pass through activation
        a = np.maximum(a,0).T
        all_mean = np.reshape(np.mean(a, axis=1), (np.shape(a)[0],1))
        all_std = np.reshape(np.std(a,axis=1), (np.shape(a)[0],1))
        a = a - all_mean
        a = a/(all_std+10**-15)
    
    return a

def feature_transforms_wo(a, w):
    # loop through each layer
    for W in w:
    # compute inner -product with current layer weights
        a = W[0] + np. dot(a.T , W[1:])        
        # pass through activation
        a = np.tanh(a). T
    return a

# neural network model
def model(theta,x,y,batch_inds):
    # print(np.size(batch_inds))
    y = np.reshape(y,(1, np.size(y)))
    # print('Again the size of x', np.shape(x))
    # print('Again the size of y', np.size(y))
    x_p = x[:,batch_inds]
    y_p = y[:,batch_inds]
    # print('size of y_p is', np.shape(y_p))
    # compute feature transformation
    f = feature_transforms(x_p, theta[0])

    # compute final linear combination
    a = theta[1][0] + np. dot(f.T, theta [1][1:])
    # a = a.T
    # # print('shape of a is:', np.shape(a))
    # all_mean = np.reshape(np.mean(a, axis=1), (np.shape(a)[0],1))
    # # print('shape of mean is', np.shape(all_mean))
    # all_std = np.reshape(np.std(a,axis=1), (np.shape(a)[0],1))
    # a = a - all_mean
    # a = a/all_std
    # a = a.T
    cost = multisoftmax(a, y_p)
    
    return cost

def model_wo(theta,x,y,batch_inds):
    # print(np.size(batch_inds))
    y = np.reshape(y,(1, np.size(y)))
    # print('Again the size of x', np.shape(x))
    # print('Again the size of y', np.size(y))
    x_p = x[:,batch_inds]
    y_p = y[:,batch_inds]
    # print('size of y_p is', np.shape(y_p))
    # compute feature transformation
    f = feature_transforms_wo(x_p, theta[0])

    # compute final linear combination
    a = theta[1][0] + np. dot(f.T, theta [1][1:])
    cost = multisoftmax(a, y_p)
    return cost

def prediction(theta):
    # compute feature transformation
    f = feature_transforms(x, theta[0])

    # compute final linear combination
    a = theta[1][0] + np. dot(f.T, theta [1][1:])

    return a

def prediction_wo(theta):
    # compute feature transformation
    f = feature_transforms_wo(x, theta[0])

    # compute final linear combination
    a = theta[1][0] + np. dot(f.T, theta [1][1:])

    return a
    
# a =  model(theta_init)
# # print('Size of a is', np.shape(a))

max_its = 100
alpha_choice = 0.1
batch_size = 600
# weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,theta_init)
weight_history,cost_history = gradient_descent_batch(model,theta_init,x,y,alpha_choice,max_its,batch_size)
print(cost_history)

# theta_init = network_initializer(layer_sizes, scale)
weight_history_wo,cost_history_wo = gradient_descent_batch(model_wo,theta_init,x,y,alpha_choice,max_its,batch_size)
print(cost_history_wo)

# print('shape of weight_history is:', np.shape(weight_history))

misclassification = np.zeros((np.size(cost_history),1))
misclassification_wo = np.zeros((np.size(cost_history_wo),1))

# for i in range(1):
for i in range(np.size(cost_history)):
    pred = prediction(weight_history[i])
    pred_wo = prediction_wo(weight_history_wo[i])
    # print('size of pred is:', np.shape(pred))
    pred = np.reshape(np.argmax(pred, axis=1), (1,np.size(y)))
    pred_wo = np.reshape(np.argmax(pred_wo, axis=1), (1,np.size(y)))

    # print('checking pred again', np.)
    # print('Shape oy pred is', np.shape(pred))
    pred = (pred == y.T)
    pred_wo = (pred_wo == y.T)
    misclassification[i] = np.size(y) - np.sum(pred)
    misclassification_wo[i] = np.size(y) - np.sum(pred_wo)
    # print(misclassification[i])

# print(misclassification)
# print('At the end of the analysis, the # of misclassifications are:', misclassification[-1])
# print('End weight is:', weight_history[-1])





plt.figure(1)
plt.plot(cost_history)
plt.plot(cost_history_wo)
plt.legend(["With batch normalization", "No batch normalization"])
plt.title('Cost function history')

plt.figure(2)
plt.plot(misclassification)
plt.plot(misclassification_wo)
plt.legend(["With batch normalization", "No batch normalization"])
plt.title('# misclassification history')

plt.show()
