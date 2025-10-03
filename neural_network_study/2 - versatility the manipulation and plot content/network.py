
import numpy as np



def one_neuronium_simple(inputs, pesos, bias, activation_func):
    soma = np.dot(inputs, pesos) + bias
    return activation_func(soma)

import numpy as np

def train_perceptron(x,
                     y,
                     activation_function,
                     learning_rate=0.1,
                     epochs=20,
                     verbose=True):
    """
    Treina um perceptron simples (um único neurônio) e mostra progresso se verbose=True.

    x: entradas (array NxM)
    y: saídas esperadas (array Nx1)
    activation_function: função de ativação do neurônio
    learning_rate: taxa de aprendizado
    epochs: número de épocas de treino
    verbose: se True, imprime progresso
    """
    n_features = x.shape[1]
    pesos = np.random.rand(n_features)
    bias = np.random.rand(1)

    for epoca in range(epochs):
        if verbose:
            print(f"\nÉpoca {epoca+1}/{epochs}")
        for i in range(len(x)):
            entrada = x[i]
            esperado = y[i]

            # saída do neurônio
            soma = np.dot(entrada, pesos) + bias
            obtido = activation_function(soma)

            # erro
            erro = esperado - obtido

            # atualização dos pesos e bias
            pesos += learning_rate * erro * entrada
            bias += learning_rate * erro

            if verbose:
                print(f"Entrada: {entrada}, Esperado: {esperado}, Obtido: {obtido}, "
                      f"Pesos: {pesos}, Bias: {bias}")

    return pesos, bias



