import numpy as np
from .neural_network import NeuralNetwork
from .error import ErrorFunctions

class Optimizer():
    """
    Class for the optimizers based on two different algorithms

    RMSProp()
    AdamW() 
    """
    def __init__(self,lr:float,maxEpochs:int,goal:float,mingrad:float,nn: NeuralNetwork,
                 inputs:np.array,targets:np.array,error_fun,show:int =1,consecutive_epochs:int =10,
                 batch_size: int=1)->None:  
        self.nn = nn
        self.name = "DEFAULT"
        self.lr = lr
        self.batch_size = batch_size if batch_size > 0 else inputs.shape[0]  # Si batch_size <= 0, usa todos los ejemplos
        self.maxEpochs = maxEpochs
        self.goal = goal
        self.mingrad = mingrad
        self.show = show
        self.inputs = inputs
        self.targets = targets
        self.error_fun = error_fun
        self.consecutive_epochs = consecutive_epochs
        
        
    def optimize(self):
        this = self.name
        stop = ""
        epochs = []
        perfs  = []
        consecutive_rise = 0  # Contador para el número de épocas consecutivas en las que el rendimiento ha subido
        prev_perf = float('inf')
        num_samples = self.inputs.shape[0]  # Número de ejemplos de entrenamiento
        print("\n")

        #Entrenamiento
        for epoch in range(self.maxEpochs+1):
            #Mezclar los datos
            permutation = np.random.permutation(num_samples)
            inputs_shuffled = self.inputs[permutation, :]
            targets_shuffled = self.targets[permutation, :]

            # Procesar mini-lotes
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_inputs = inputs_shuffled[start:end,: ]
                batch_targets = targets_shuffled[start:end,:]

                # Performance and Gradient
                _ = self.nn.forwardPass(batch_inputs)
                gX = self.nn.backwardPass(batch_targets)

                # Aplanar y concatenar los gradientes en un solo vector
                gX_flattened = np.concatenate([grad.flatten() for grad in gX])
                  
                self.train(gX_flattened)  # Pasar gX aplanado
                
            if self.batch_size != self.inputs.shape[0]:
                _ = self.nn.forwardPass(self.inputs)
                gX = self.nn.backwardPass(self.targets)
                
            perf = self.nn.error(self.targets, self.error_fun)
            
            # Aplanar y concatenar los gradientes en un solo vector
            gX_flattened = np.concatenate([grad.flatten() for grad in gX])
            normgX = np.linalg.norm(gX_flattened)

            # Stopping criteria
            epochs = np.append(epochs, epoch)
            perfs = np.append(perfs, perf)
            if np.all(perf <= self.goal):
                stop = "Performance goal met"
            elif epoch == self.maxEpochs:
                stop = "Maximum epoch reached, performance goal was not met"
            elif normgX < self.mingrad:
                stop = "Minimum gradient reached, performance goal was not met"
            elif perf >= prev_perf or (abs(perf - prev_perf) < self.goal * 10):
                consecutive_rise += 1
                if consecutive_rise >= self.consecutive_epochs:
                    stop = f"Performance has risen for {self.consecutive_epochs} consecutive epochs"
            elif perf < prev_perf:
                consecutive_rise = 0

            prev_perf = perf
            if (np.fmod(epoch, self.show) == 0 or len(stop) != 0):
                print(this, end=": ")
                if np.isfinite(self.maxEpochs):
                    print("Epoch ", epoch, "/", self.maxEpochs, end=" ")
                if np.isfinite(self.goal):
                    print(", Performance %8.3e" % perf, "/", self.goal, end=" ")
                if np.isfinite(self.mingrad):
                    print(", Gradient %8.3e" % normgX, "/", self.mingrad)

                if len(stop) != 0:
                    print("\n", this, ":", stop, "\n")
                    break            
        return perfs, epochs

    def train(self,gX):
        raise NotImplementedError("No se ha definido el optimizador, esta es la clase base")
    
    
class RmsProp(Optimizer):
    def __init__(self, nn: NeuralNetwork, inputs:np.array, targets:np.array,lr: float =1e-3, batch_size: int =0, maxEpochs: int =500, 
                 goal: float =1e-8,mingrad: float =1e-11, show:int =1, error_fun=ErrorFunctions.SSE, 
                 consecutive_epochs: int=10,WDecay:float=0,alpha:float=0.99,centered:bool=False,
                 momentum:float=0.6,epsilon:float=1e-9) -> None:
        if not isinstance(inputs, np.ndarray):
            raise TypeError("El argumento 'inputs' debe ser un array de NumPy.")
        if not isinstance(targets, np.ndarray):
            raise TypeError("El argumento 'targets' debe ser un array de NumPy.")
        super().__init__(lr,maxEpochs,goal,mingrad,nn,inputs,targets,error_fun,show,consecutive_epochs,batch_size)
        self.name = "trainRMSPROP"
        self.epsilon = epsilon
        self.v = np.zeros_like(np.concatenate([w.flatten() for w in nn.weights]))  # Vector de acumulación de gradientes
        self.vh = 0
        self.b = 0
        self.gAvg = 0
        self.WDecay = WDecay
        self.alpha = alpha
        self.centered = centered
        self.momentum = momentum

    def train(self, gX):        
        if self.WDecay != 0:
            gX = gX + gX*self.WDecay
        self.v = self.alpha*self.v + ((1-self.alpha)*(gX**2))
        self.vh = self.v
        if self.centered:
            self.gAvg = self.gAvg*self.alpha + ((1-self.alpha)*gX)
            self.vh = self.vh - self.gAvg**2
        if self.momentum > 0:
            self.b = self.momentum*self.b + gX/((self.vh**(1/2))+self.epsilon)
            update = self.lr*self.b
        else:
            update = self.lr*(gX/((self.vh**(1/2))+1e-8))
                
        # Actualizar pesos
        start = 0
        for i, w in enumerate(self.nn.weights):
            shape = w.shape
            size = np.prod(shape)
            grad_update = update[start:start+size].reshape(shape)
            #Aplicar la penalizacion por regularizacion si es necesario
            if self.nn.regularization == 'L1':
                reg_penalty = self.nn.lambda_reg * np.sign(w)
                grad_update += reg_penalty
            elif self.nn.regularization == 'L2':
                reg_penalty = self.nn.lambda_reg * 2 * w
                grad_update += reg_penalty    
                
            self.nn.weights[i] -= grad_update
            start += size
            
            
class AdamW(Optimizer):
    def __init__(self, nn: NeuralNetwork, inputs:np.array, targets:np.array,lr: float =1e-3, batch_size: int =0, maxEpochs: int =500, 
                 goal: float =1e-8,mingrad: float =1e-11, show:int =1, error_fun=ErrorFunctions.SSE, 
                 consecutive_epochs: int=10,WDecay:float=0.001,maximize:bool=False,amsgrad:bool=False,
                 b1:float=0.9,b2:float=0.999,epsilon:float=1e-9) -> None:
        if not isinstance(inputs, np.ndarray):
            raise TypeError("El argumento 'inputs' debe ser un array de NumPy.")
        if not isinstance(targets, np.ndarray):
            raise TypeError("El argumento 'targets' debe ser un array de NumPy.")
        super().__init__(lr,maxEpochs,goal,mingrad,nn,inputs,targets,error_fun,show,consecutive_epochs,batch_size)
        self.name = "trainAdamW"
        self.WDecay = WDecay
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.m = [np.zeros_like(w) for w in nn.weights]
        self.v = [np.zeros_like(w) for w in nn.weights]
        self.vMax = [np.zeros_like(w) for w in nn.weights] if amsgrad else None
        self.epoch = 0
    
    def optimize(self):
        perfs,epochs = super().optimize()
        self.epoch = 0 #Reseteamos esto en caso se necesite optimizar de nuevo sin crear un nuevo optimizador
        return perfs,epochs
        
    def train(self, gX):
        if self.maximize:
            gX = [-gx for gx in gX]

        # Weight decay (L2 regularization)
        for i in range(len(self.nn.weights)):
            self.nn.weights[i] -= self.lr * self.WDecay * self.nn.weights[i]

        # Update biased first moment estimate
        self.m = [self.b1 * m + (1 - self.b1) * gx for m, gx in zip(self.m, gX)]
        # Update biased second raw moment estimate
        self.v = [self.b2 * v + (1 - self.b2) * (gx ** 2) for v, gx in zip(self.v, gX)]
        
        # Compute bias-corrected first moment estimate
        m_hat = [m / (1 - self.b1 ** (self.epoch + 1)) for m in self.m]
        # Compute bias-corrected second raw moment estimate
        v_hat = [v / (1 - self.b2 ** (self.epoch + 1)) for v in self.v]

        if self.amsgrad:
            self.vMax = [np.maximum(vm, vh) for vm, vh in zip(self.vMax, v_hat)]
            v_hat = self.vMax
        
        # Compute the update
        v_hat_sqrt = [np.sqrt(vh) + self.epsilon for vh in v_hat]
        updates = [self.lr * mh / vh_sqrt for mh, vh_sqrt in zip(m_hat, v_hat_sqrt)]

        # Apply the update
        for i in range(len(self.nn.weights)):
            self.nn.weights[i] -= updates[i]

            # Aplicar la penalización por regularización si es necesario
            if self.nn.regularization == 'L1':
                reg_penalty = self.nn.lambda_reg * np.sign(self.nn.weights[i])
                self.nn.weights[i] -= self.lr * reg_penalty
            elif self.nn.regularization == 'L2':
                reg_penalty = self.nn.lambda_reg * 2 * self.nn.weights[i]
                self.nn.weights[i] -= self.lr * reg_penalty

        self.epoch += 1