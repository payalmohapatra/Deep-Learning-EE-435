from numpy.core.fromnumeric import argmax
#import mnist
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from skimage import exposure
from sklearn.datasets import fetch_openml
#from optimizers import gradient_descent
from optimizers_only import gradient_descent

w = 2.0
## cost function
def model(w):
    g = (w**4 + w**2 + 10*w)*(1/50)
    return g
# g = lambda w: (w*w*w*w + w*w + 10.0*w)/50.0

plt.figure(1)
plt.legend(["g(w)", "g'(w)"])

max_its = 1000
alpha_choice = 1
weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

alpha_choice = 0.1
weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

alpha_choice = 0.001
weight_history,cost_history = gradient_descent(model ,alpha_choice,max_its,w)
plt.plot(cost_history)

plt.xlabel('Iterations')
plt.ylabel('Cost function (g(w))')
plt.legend(["alpha = 1", "alpha = 0.1", "alpha = 0.001"])
plt.title('Cost functions with alpha as 1, 0.1, 0.001.')
# plt.show()

### plotting g(w) and 
f = lambda w: (w**4 + w**2 + 10.0*w)/50.0
deri = lambda w: (4*w**3 + 2*w + 10.0)/50.0
plt.figure(2)
xpts = np.linspace(-5, 5, 50)
plt.plot(xpts, f(xpts))
plt.plot(xpts, deri(xpts))
plt.legend(["g(w)", "g'(w)"])
plt.xlabel('w')
plt.title('Plot of the function g(w) and its derivative')
plt.show()
