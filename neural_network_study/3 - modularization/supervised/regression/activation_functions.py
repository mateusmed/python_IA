import matplotlib.pyplot as plt
import numpy as np
import math

# Definições das funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
    output = valor já ativado pela sigmoid, não o valor bruto (z)
"""
def sigmoid_derivative(output):
    return output * (1.0 - output)

def relu(x):
    return np.maximum(0, x)

def step_activation(x):
    return np.where(x > 0, 1, 0)

def linear(x):
    """Função de ativação linear (identidade). Retorna o próprio valor."""
    return x

def linear_derivative(output):
    """Derivada da função linear. É sempre 1."""
    return 1


# plot_grafic()
