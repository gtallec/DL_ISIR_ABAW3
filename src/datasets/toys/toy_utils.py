import numpy as np

def bernouilli(p):
    U = np.random.random(size=p.shape)
    return (U - p <= 0).astype(int)

def T_sigmoid(T):
    def fun(X):
        return 1/(1 + np.exp(-X/T))
    return fun

def trunc(x, decimals):
    x_str = str(x).split('.')
    if len(x_str) == 1:
        return x_str[0]
    return x_str[0] + '.' + x_str[1][:decimals]


