"""
Este script implementa um perceptron simples, que é um tipo de rede neural artificial.
Ele recebe duas entradas, pesos e um viés, realiza a soma ponderada das entradas, e usa uma
função de ativação do tipo step para gerar a saída.
"""

import math

def step_activation(x):
    return 1 if x > 0 else 0

def perceptron(x1, x2, peso1, peso2, bias):
    soma = (x1 * peso1) + (x2 * peso2) + bias
    return step_activation(soma)


entrada1 = 1
entrada2 = 1
peso1 = 1.0
peso2 = 1.0
bias = -1.5


saida = perceptron(entrada1, entrada2, peso1, peso2, bias)
print(f"Saída para as entradas {entrada1}, {entrada2}: {saida}")
