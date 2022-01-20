from numpy.core.fromnumeric import argmax
#import mnist
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
#from optimizers import gradient_descent
from optimizers_only import gradient_descent
from optimizers_only import gradient_descent_momentum
from optimizers_only import gradient_descent_component
from optimizers_only import gradient_descent_full_norm

w = np.array([[2.0], [2.0]])
# b = np.array([[0.0], [1.0]])
C = np.array([[0.5, 0],[0, 9.75]])

## cost function
def model(w):
    g = max(0, np.tanh(4*w[0] + 4*w[1])) + np.abs(0.4*w[0]) + 1
    return g
# g = lambda w: (w*w*w*w + w*w + 10.0*w)/50.0

plt.figure(1)
plt.legend(["g(w)", "g'(w)"])

max_its =100
alpha_choice = 0.1
weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

weight_history,cost_history = gradient_descent_component(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

weight_history,cost_history = gradient_descent_full_norm(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

plt.xlabel('Iterations')
plt.ylabel('Cost function (g(w))')
plt.legend(["gradient descent","component-wise normalization","Fully normalized"])
plt.show()