import numpy as np

class ErrorFunctions:
    @staticmethod
    def MSE(y_true, y_pred):
        """
        Calculate the mean squared error between true and predicted values.

        Parameters:
        y_true: numpy.ndarray
            True values
        y_pred: numpy.ndarray
            Predicted values

        Returns:
        float
            Mean squared error
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def MAE(y_true, y_pred):
        """
        Calculate the mean absolute error between true and predicted values.

        Parameters:
        y_true: numpy.ndarray
            True values
        y_pred: numpy.ndarray
            Predicted values

        Returns:
        float
            Mean absolute error
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def SSE(y_true, y_pred):
        """
        Calculate the sum of squared errors between true and predicted values.

        Parameters:
        y_true: numpy.ndarray
            True values
        y_pred: numpy.ndarray
            Predicted values

        Returns:
        float
            Sum of squared errors
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sum((y_true - y_pred) ** 2)
    
    @staticmethod
    def RMSE(y_true, y_pred):
        """
        Calculate the root mean squared error between true and predicted values.

        Parameters:
        y_true: numpy.ndarray
            True values
        y_pred: numpy.ndarray
            Predicted values

        Returns:
        float
            Root mean squared error
        """
        return np.sqrt(ErrorFunctions.MSE(y_true, y_pred))
    
    def R2(y_true, y_pred):
        
        sum_squares_residuals = sum((y_true - y_pred) ** 2)
        sum_squares = sum((y_true - np.mean(y_true)) ** 2)
        return 1 - sum_squares_residuals / sum_squares