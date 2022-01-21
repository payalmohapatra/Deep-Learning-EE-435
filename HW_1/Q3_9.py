""" Contact : PayalMohapatra2026@u.northwestern.edu
Sources :
https://github.com/jermwatt/machine_learning_refined/tree/gh-pages/mlrefined_libraries
"""
from numpy.core.fromnumeric import argmax
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
#from optimizers import gradient_descent
from optimizers_only import gradient_descent
from optimizers_only import gradient_descent_momentum

w = np.array([[10.0], [1.0]])
C = np.array([[0.5, 0],[0, 9.75]])

## cost function
def model(w):
    g = np.dot(np.dot(w.T, C), w)
    return g
# g = lambda w: (w*w*w*w + w*w + 10.0*w)/50.0

plt.figure(1)
plt.legend(["g(w)", "g'(w)"])

max_its =25
alpha_choice = 0.1
weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)


## momentum accelerated
beta = 0.2
weight_history,cost_history = gradient_descent_momentum(model ,alpha_choice,max_its,w, beta)
plt.plot(cost_history)

beta = 0.7
weight_history,cost_history = gradient_descent_momentum(model ,alpha_choice,max_its,w, beta)
plt.plot(cost_history)

# alpha_choice = 0.1
# weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
# plt.plot(cost_history)

# alpha_choice = 0.001
# weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
# plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost function (g(w))')
plt.legend(["standard gradient descent","moment accelerated gradient descent with beta = 0.2","moment accelerated gradient descent with beta=0.7"])
plt.title('Cost functions history plot with standard and moment accelerated gradient descent optimisation.')
plt.show()

#Finding :: Anomaly at point 1 since only one point to average