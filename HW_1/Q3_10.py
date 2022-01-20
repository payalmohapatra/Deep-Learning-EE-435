from numpy.core.fromnumeric import argmax
#import mnist
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
#from optimizers import gradient_descent
from optimizers_only import gradient_descent
from optimizers_only import gradient_descent_full_norm

w = np.array([[2.0],[2.0]])

## cost function
# g(w1, w2) = tanh(4 w1 + 4 w2) + max(1, 0.4 w21) + 1.
def model(w): # w is a (1,2) vector
    g = np.tanh(4.0*w[0] + 4.0*w[1]) + max(1.0, 0.4*w[0]*w[0]) + 1.0 
    return g


max_its =1000
alpha_choice = 0.1
weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)


## full normalised
weight_history,cost_history = gradient_descent_full_norm(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

plt.legend(["Gradient Descent","Full Normalised Gradient Descent"])
plt.show()