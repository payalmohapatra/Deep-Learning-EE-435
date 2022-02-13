import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from skimage import exposure
from sklearn.datasets import fetch_openml
from optimizers_only import gradient_descent_nn
from optimizers_only import gradient_descent
import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func


# import data
datapath = 'Data/'
X = np.loadtxt(datapath + 'universal_autoencoder_samples.txt', delimiter=',')

plt.scatter(X[0,:], X[1,:], c = 'k', s = 60, linewidth = 0.75, edgecolor = 'w')
#plt.show()

x = X[0,:]
y = X[1,:]

# Normalise inputs
#for i in range(np.size(x,0)):    
#    row_mean = np.mean(x[i,:])
#    row_std = np.std(x[i,:])
#    x[i,:] = x[i,:]-row_mean/row_std 
##x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# neural network feature transformation
def feature_transforms(a, w):
    # loop through each layer
    for W in w:
        # compute inner -product with current layer weigh
        a = W[0] + np. dot(a.T , W[1:])
        # pass through activation
        a = np.tanh(a). T
    return a

# neural network model
def model(x, theta):
#def model(theta):
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

   
   
def rmse_func(x,w) :
    cost = 0
    y_pred = model(x,w)
    #y_pred = y_pred._value
    for p in range(y.size):
       # get pth prediction/ output pair
       y_p = y[:,p]
       y_model = y_pred[:,p]
       ## add to current cost
       cost += (y_p - y_model)**2
    return cost/ float(np.size(y))

def gradient_descent_nn(g,alpha_choice,max_its,w,x):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(
        else:
            alpha = alpha_choice
        
        # evaluate the gradient, store current (unflattened) weights and cost function value
        cost_eval,grad_eval = gradient(x,w)
        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

        # take gradient descent step
        w = w - alpha*grad_eval
            
    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing 
    # the gradient at the final step we don't get the final cost function value 
    # via the Automatic Differentiatoor) 
    cost_history.append(g_flat(w))  
    return weight_history,cost_history


# Initialise as per Example 13.4 from text
layer_sizes = [np.shape(x)[0],10,10,10,1]
w_init = network_initializer(layer_sizes,1.0)
max_its = 2000
alpha_choice = 0.1
weight_history,cost_history = gradient_descent_nn(rmse_func,alpha_choice,max_its,w_init,x)
#weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w_init)
plt.plot(cost_history)
plt.title('Cost history vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.figure()
#plt.show()

#y_p = model(weight_history[-1])
#print(np.shape(y_p))
#plt.scatter(x[0,:],y_p[0,:], s=30, c='b', marker="s", label='model fit')
#plt.scatter(x[0,:],y[0,:], s=30, c='r', marker="o", label='original')
#plt.legend(loc='upper left')
#plt.show()