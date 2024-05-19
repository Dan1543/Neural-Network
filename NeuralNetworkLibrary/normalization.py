import numpy as np
def printData(inputs,targets):
    # Calcular estadísticas descriptivas para los datos de entrada
    mean_inputs = np.mean(inputs, axis=0)
    std_inputs = np.std(inputs, axis=0)
    min_inputs = np.min(inputs, axis=0)
    max_inputs = np.max(inputs, axis=0)

    # Calcular estadísticas descriptivas para los datos de salida (targets)
    mean_targets = np.mean(targets, axis=0)
    std_targets = np.std(targets, axis=0)
    min_targets = np.min(targets, axis=0)
    max_targets = np.max(targets, axis=0)

    # Imprimir las estadísticas
    print("Estadísticas de los datos de entrada:")
    print("Media:", mean_inputs)
    print("Desviación estándar:", std_inputs)
    print("Mínimo:", min_inputs)
    print("Máximo:", max_inputs)

    print("\nEstadísticas de los datos de salida:")
    print("Media:", mean_targets)
    print("Desviación estándar:", std_targets)
    print("Mínimo:", min_targets)
    print("Máximo:", max_targets)
    
def min_max(inputs,targets):
    inputs_min = np.min(inputs, axis=0, keepdims=True)
    inputs_max = np.max(inputs, axis=0, keepdims=True)
    inputs_normalized = (inputs - inputs_min) / (inputs_max - inputs_min)
    
    targets_min = np.min(targets, axis=0, keepdims=True)
    targets_max = np.max(targets, axis=0, keepdims=True)
    targets_normalized = (targets - targets_min) / (targets_max - targets_min)
    return inputs_normalized,targets_normalized