"""
Este script demonstra o funcionamento de portas lógicas básicas (AND, OR, NAND, NOR, XOR)
utilizando diferentes funções de ativação (sigmoide, ReLU e step).
Ele calcula e exibe os resultados dessas operações lógicas para entradas binárias.
"""

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def step_activation(x):
    return 1 if x > 0 else 0

def neuronio_AND(x1, x2, activation_func):
    peso1, peso2, bias = 1.0, 1.0, -1.5
    soma = (x1 * peso1) + (x2 * peso2) + bias
    return activation_func(soma)

def neuronio_OR(x1, x2, activation_func):
    peso1, peso2, bias = 1.0, 1.0, -0.5
    soma = (x1 * peso1) + (x2 * peso2) + bias
    return activation_func(soma)

def neuronio_NAND(x1, x2, activation_func):
    peso1, peso2, bias = -1.0, -1.0, 1.5
    soma = (x1 * peso1) + (x2 * peso2) + bias
    return activation_func(soma)

def neuronio_NOR(x1, x2, activation_func):
    peso1, peso2, bias = -1.0, -1.0, 0.5
    soma = (x1 * peso1) + (x2 * peso2) + bias
    return activation_func(soma)

def neuronio_XOR(x1, x2, activation_func):
    return neuronio_NAND(neuronio_OR(x1, x2, activation_func), neuronio_AND(x1, x2, activation_func), activation_func)


print("AND (Sigmoide): ", neuronio_AND(1, 1, sigmoid))
print("AND (ReLU): ", neuronio_AND(1, 1, relu))
print("AND (Step): ", neuronio_AND(1, 1, step_activation))

print("OR (Sigmoide): ", neuronio_OR(1, 1, sigmoid))
print("OR (ReLU): ", neuronio_OR(1, 1, relu))
print("OR (Step): ", neuronio_OR(1, 1, step_activation))

print("NAND (Sigmoide): ", neuronio_NAND(1, 1, sigmoid))
print("NAND (ReLU): ", neuronio_NAND(1, 1, relu))
print("NAND (Step): ", neuronio_NAND(1, 1, step_activation))

print("NOR (Sigmoide): ", neuronio_NOR(1, 1, sigmoid))
print("NOR (ReLU): ", neuronio_NOR(1, 1, relu))
print("NOR (Step): ", neuronio_NOR(1, 1, step_activation))

print("XOR (Sigmoide): ", neuronio_XOR(1, 1, sigmoid))
print("XOR (ReLU): ", neuronio_XOR(1, 1, relu))
print("XOR (Step): ", neuronio_XOR(1, 1, step_activation))
