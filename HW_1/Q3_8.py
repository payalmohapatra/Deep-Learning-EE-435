from numpy.core.fromnumeric import argmax
#import mnist
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from skimage import exposure
from sklearn.datasets import fetch_openml
#from optimizers import gradient_descent
from optimizers_only import gradient_descent

w = np.zeros((10,1)) + 0.5
## cost function
def model(w):
    g = np.dot(w.T,w)
    return g

max_its = 100
alpha_choice = 1
weight_history_1,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

alpha_choice = 0.1
weight_history_0_1,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

alpha_choice = 0.001
weight_history_0_001,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

plt.xlabel('Iterations')
plt.ylabel('Cost function (g(w))')
plt.legend(["alpha = 1", "alpha = 0.1", "alpha = 0.001"])
plt.title('Cost functions with alpha as 1, 0.1, 0.001.')
plt.show()

print(weight_history_0_001[-1])
print(weight_history_0_1[-1])
print(weight_history_1[-1])

# Findings ::
# for alpha = 1 the w_update does not change because the cost function is giving same value for positive and negative values of w.