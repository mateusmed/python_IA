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



# plot_grafic()