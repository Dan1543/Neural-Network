from scipy.special import erf
import numpy as np

class activationFunction:
    """
    Class for activation functions meant to be used on Neural Networks
    All the activation functions and its derivatives are available on the subclases
    Subclasses:
    Swish()
    Relu()
    Purelin()
    Logsig()
    Tansig()
    Radbas()
    Tribas()
    RadBasN()
    HardLim()
    HardLims()
    SatLin()
    SatLins()
    Softmax()
    LeakyRelu()
    ELU()
    GELU()
    PReLU()
    SELU()
    SiLU()
    Softplus()
    """
    def function(self,x):
        """Activation function"""
        raise NotImplementedError("This is only the base function, the implementation of this is on any of the other functions, for more information check the class DOCSTRING")
    def derivative(self,x):
        """Derivative of the activation function"""
        raise NotImplementedError("This is only the base function, the implementation of this is on any of the other functions, for more information check the class DOCSTRING")
    def active(self):
        raise NotImplementedError("This is only the base function, the implementation of this is on any of the other functions, for more information check the class DOCSTRING")

class Swish(activationFunction):
    """Scaled Exponential Linear Unit With a Shift function"""
    def __init__(self, beta=1):
        self.beta = beta
    def function(self, x):
        return x * (1 / (1 + np.exp(-self.beta * x)))
    def derivative(self, x):
        return (self.beta * self.function(x)) + (1 / (1 + np.exp(-self.beta * x))) * (1 - self.beta * self.function(x))
    def active(self):
        out = [-float('inf'), float('inf')]
        return out
    
class Relu(activationFunction):
    """Rectified linear unit function (ReLU)"""
    def function(self, x):
        return np.maximum(0,x)
    def derivative(self, x):
        return np.where(x>0,1,0)
    def active(self):
        out = [0, float('inf')]
        return out
    
class Purelin(activationFunction):
    """Linear (Identity) function"""
    def function(self,x):
      return x
    def derivative(self,x):
      return np.ones_like(x)
    def active(self):
        out = [-float('inf'), float('inf')]
        return out

class Logsig(activationFunction):
    """Logistic function"""
    def function(self, x):
      return 1 / (1 + np.exp(-x))
    def derivative(self,x):
        return self.function(x) * (1 - self.function(x))
    def active(self):
        out = [-4.0, 4.0]
        return out
    
class Tansig(activationFunction):
    """Hyperbolic function"""
    def function(self,x):
        return np.tanh(x)
    def derivative(self,x):
        return  1- np.tanh(x)**2
    def active(self):
        out = [-2, 2]
        return out

class Radbas(activationFunction):
    """Gaussian function"""
    def function(self,x):
        return np.exp(-x**2)
    def derivative(self,x):
        return -2 * x * np.exp(-x**2)
    def active(self):
        out = [-2, 2]
        return out

class Tribas(activationFunction):
    """Triangular basis function"""
    def function(self, x):
      return np.maximum(0, 1 - np.abs(x))
    def derivative(self, x):
      return np.where(np.abs(x) < 1, -1, 0)
    def active(self):
        out = [-1, 1]
        return out
    
class RadBasN(activationFunction):
    """Normalized radial basis function"""
    def __init__(self, sigma=1):
        """
        PARAMETERS
        sigma : float by default 1
        """
        self.sigma = sigma

    def function(self, x):
      return np.exp(-0.5 * (x / self.sigma)**2)
    def derivative(self, x):
      return -x / self.sigma**2 * np.exp(-0.5 * (x / self.sigma)**2)
    def active(self):
        out = [-2, 2]
        return out

class HardLim(activationFunction):
    """Hard limit function"""
    def function(self, x):
        return np.where(x >= 0, 1, 0)
    def derivative(self, x):
        return np.zeros_like(x)
    def active(self):
        out = [0, 0]
        return out
    
class HardLims(activationFunction):
    """Symmetric hard limit function"""
    def function(self, x):
        return np.where(x >= 0, 1, -1)
    def derivative(self, x):
        return np.zeros_like(x)
    def active(self):
        out = [0, 0]
        return out

class SatLin(activationFunction):
    """Saturatin linear function"""
    def function(self, x):
        return np.clip(x, 0, None)

    def derivative(self, x):
        return np.where(x >= 0, 1, 0)
    
    def active(self):
        out = [-0, 1]
        return out
    
class SatLins(activationFunction):
    """Symmetric saturating function"""
    def function(self, x):
        return np.clip(x, -1, 1)

    def derivative(self, x):
        return np.where(np.logical_and(x >= -1, x <= 1), 1, 0)
    
    def active(self):
        out = [-1, 1]
        return out
    
class Softmax(activationFunction):
    """Normalized exponential function (softmax)"""
    def function(self, x):
        x  = np.subtract(x, np.max(x))        # prevent overflow
        ex = np.exp(x)
        return ex / np.sum(ex)
    
    def derivative(self, x):
        raise NotImplementedError("La derivada de Softmax no se utiliza típicamente en el entrenamiento de redes neuronales.")
    
    def active(self):
        out = [-float('inf'), float('inf')]
        return out
    
class LeakyRelu(activationFunction):
    """Leaky rectified linear unit function (leakyRelu)"""
    def function(self, x):
        return np.where(x>0,x,1e-2*x)
    def derivative(self, x):
        return np.where(x>0,1,1e-2)
    def active(self):
        out = [-float('inf'), float('inf')]
        return out
    
class ELU(activationFunction):
    """Exponential Linear Unit function (ELU)"""
    def __init__(self, alpha=1):
        """
        PARAMETERS:
        alpha = float by default 1
        """
        self.alpha=alpha
    def function(self, x):
        return np.where(x>0,x,self.alpha*(np.exp(x)-1))
    def derivative(self, x):
        return np.where(x>0,1,self.alpha*np.exp(x))
    def active(self):
        out = [-float('inf'), float('inf')]
        return out
    
class GELU(activationFunction):
    """Gaussian Error Linear Unit function (GELU)"""
    def function(self, x):
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    def derivative(self, x):
        return 0.5 * (1 + erf(x / np.sqrt(2))) + (x / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    def active(self):
        out = [-float('inf'), float('inf')]
        return out
    
class PReLU(activationFunction):
    """Parametric rectified linear unit function (PReLU)"""
    def __init__(self, alpha=1e-1):
        """
        PARAMETERS
        alpha : float by default 1e-1
        """
        self.alpha=alpha
    def function(self, x):
        return np.where(x<0,self.alpha*x,x)
    def derivative(self, x):
        return np.where(x<0,self.alpha,1)
    def active(self):
        out = [-float('inf'), float('inf')]
        return out
    
class SELU(activationFunction):
    """Scaled exponential linear unit function (SELU)"""
    def __init__(self, lamb= 1.0507, alpha=1.67326):
        """
        PARAMETERS
        lamb : float by default 1.0507
        alpha : float by default 1.67326
        Both are suposed to be always that value so it's recomended to not change them
        """
        self.lamb=lamb
        self.alpha=alpha
    def function(self, x):
        return self.lamb * np.where(x<0, self.alpha*(np.exp(x)-1),x)
    def derivative(self, x):
        return self.lamb * np.where(x<0, self.alpha*np.exp(x),1)
    def active(self):
        out = [-float('inf'), float('inf')]
        return out

class SiLU(activationFunction):
    """Sigmoid linear unit function (SiLU)"""
    def function(self, x):
        return (x / (1 + np.exp(-x)))
    def derivative(self, x):
        return (1 + np.exp(-x) + x*np.exp(-x))/((1+np.exp(-x))**2)
    def active(self):
        out = [-float('inf'), float('inf')]
        return out

class Softplus(activationFunction):
    """Smooth approximation ReLU function"""
    def function(self, x):
        return np.log(1 + np.exp(x))
    def derivative(self, x):
        return 1 / (1+np.exp(-x))
    def active(self):
        out = [-float('inf'), float('inf')]
        return out