import numpy as np
from .activation import activationFunction

class NeuralNetwork:
    """Class for the structure of a Neural Network"""
    def __init__(self, input_size:int, layer_sizes:list[int], output_size:int, 
                 activation_funcs:list['activationFunction'], wInit:str='random',
                 dropout_rate:float=0, regularization:str='None',lambda_reg:float=0.01)->None:
        """
        Parameters:
        input_size: int 
            Defines the size of the input layer
        layer_sizes: int array 
            Defines the sizes of the ocult layers
        output_size: int 
            Defines the size of the output layer
        activation_funcs: activationFunction class array 
            Defines the activation function per layer
        """
        self.input_size = input_size
        self.layer_sizes = [input_size] + layer_sizes + [output_size]  # Incluir el tamaño de la capa de entrada y de salida
        self.output_size = output_size
        self.activation_funcs = activation_funcs
        self.dropout_rate = dropout_rate
        self.num_layers = len(self.layer_sizes)
        self.weights = self._initializeWeights(wInit)
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.n_outputs = []  # Lista para almacenar las salidas antes de la función de activación
        self.a_outputs = []
        self.dropout_masks = []

    
    def _initializeWeights(self, wInit):
        """
        Inicializa los pesos de la red neuronal ya sea de manera random o mediante el metodo nguyen widraw
        
        :return: Lista de matrices de pesos como np.ndarrar
        """
        weights = []
        if wInit == 'random':
            for i in range(self.num_layers - 1):
                W = np.random.randn(self.layer_sizes[i] + 1, self.layer_sizes[i+1]) # +1 para incluir los sesgos
                weights.append(W)
        elif wInit == 'nguyen':
            for i in range(self.num_layers - 1):
                ni = self.layer_sizes[i]
                no = self.layer_sizes[i+1]
                g = (0.7*no) ** (1/ni)
                active = self.activation_funcs[i].active()
                if not np.isinf(active[0]) and not np.isinf(active[1]):
                    W = np.random.randn(no, ni)
                    W = W / np.linalg.norm(W, axis=1, keepdims=True)  # Normalizacion
                    W = g * W
                    beta = np.linspace(active[0], active[1], no).reshape(-1, 1)
                    bias = g * (np.sign(W[:, 0]).reshape(-1, 1) * beta)
                    W = np.hstack([W, bias])
                    weights.append(W.T)
                else:
                    W = g * np.random.randn(ni+1,no)
                    weights.append(W)
        else:
            raise KeyError("No se reconoce el inicializador")
        return weights
        #return np.array(weights, dtype = object)
    
    def forwardPass(self, inputs, training=True):
        A = np.hstack([inputs, np.ones((inputs.shape[0], 1))])
        self.n_outputs = [inputs]
        self.a_outputs = [A]
        self.dropout_masks = []

        for i, weight in enumerate(self.weights):
            Z = np.dot(A, weight)
            self.n_outputs.append(Z)
            A = self.activation_funcs[i].function(Z)
            
            if self.dropout_rate > 0 and training and i < len(self.weights) - 1:  # Dropout en todo menos la ultima capa y solo durante el entrenamiento
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=A.shape)
                A *= dropout_mask
                self.dropout_masks.append(dropout_mask)
            elif not training and i < len(self.weights) - 1:  # Escalar datos durante la inferencia (no entrenando)
                A *= (1 - self.dropout_rate)
            
            A = np.hstack([A, np.ones((A.shape[0], 1))])
            self.a_outputs.append(A)
        return A[:, :-1]

    def backwardPass(self, targets):
        #gradients = np.array([])
        gradients = []
        e = targets - self.a_outputs[-1][:,:-1]
        ge = -2*e
        delta = ge * self.activation_funcs[-1].derivative(np.array(self.n_outputs[-1]))
        ae = self.a_outputs[-2] #El metodo forward pass deja a_outputs aumentado
        ge = np.dot(ae.T,delta)
        gradients.append(ge)
        
        for i in range(self.num_layers-2, 0, -1): 
            fdx = self.activation_funcs[i-1].derivative(np.array(self.n_outputs[i]))
            delta = fdx * np.dot(delta,self.weights[i][:-1].T)
            
            if self.dropout_rate > 0 and self.dropout_masks: #Si existe alguna mascara de dropout aplicarla
                delta *= self.dropout_masks.pop()  # Apply dropout mask
            
            ae = self.a_outputs[i-1]
            ge = np.dot(ae.T,delta)
            gradients.insert(0,ge)
        return gradients
            
    def error(self,targets,error_func):
        """
        Calculate the error based on the inputs, outputs, and error function specified.

        Parameters:
        inputs: numpy.ndarray
            Input data
        outputs: numpy.ndarray
            Output data
        error_func: function
            Error function to use (e.g., mean squared error, mean absolute error, etc.)

        Returns:
        float
            Error value calculated using the specified error function.
        """
        predicted_outputs = self.a_outputs[-1][:,:-1]
        error = error_func(targets, predicted_outputs)
        
        if self.regularization == 'L1':
            reg_term = self.lambda_reg * sum(np.sum(np.abs(w)) for w in self.weights) #Agregamos el termino de regularizacion para cada capa
        elif self.regularization == 'L2':
            reg_term = self.lambda_reg * sum(np.sum(w**2) for w in self.weights) #Agregamos el termino de regularizacion para cada capa
        else:
            reg_term = 0
        return error + reg_term #Si se usa un regularizador sera el error mas el termino del mismo de lo contrario no se agregara nada