"""
Este script implementa o treinamento de um neurônio para aprender a função XOR utilizando o algoritmo de gradiente descendente.
Ele inicializa os pesos aleatoriamente, treina o neurônio ajustando os pesos com base no erro, e testa a rede treinada.
"""


import random
import math

def step_activation(x):
    return 1 if x > 0 else 0

def calcular_erro(entrada, saida_esperada, peso1, peso2, bias):
    soma = (entrada[0] * peso1) + (entrada[1] * peso2) + bias
    saida = step_activation(soma)
    return (saida - saida_esperada) ** 2

def inicializar_pesos():
    peso1 = random.gauss(0, 1)
    peso2 = random.gauss(0, 1)
    bias = random.gauss(0, 1)
    return peso1, peso2, bias

def treinar_neuronio(entradas, saidas_esperadas, taxa_aprendizado=0.1, epocas=10000):
    peso1, peso2, bias = inicializar_pesos()

    for epoca in range(epocas):
        erro_total = 0
        for i in range(len(entradas)):
            soma = (entradas[i][0] * peso1) + (entradas[i][1] * peso2) + bias
            saida = step_activation(soma)

            erro = saidas_esperadas[i] - saida

            peso1 += taxa_aprendizado * erro * entradas[i][0]
            peso2 += taxa_aprendizado * erro * entradas[i][1]
            bias += taxa_aprendizado * erro

            erro_total += erro ** 2

        if epoca % 1000 == 0:
            print(f'Época {epoca}: Erro total = {erro_total}')

    return peso1, peso2, bias

def testar_neuronio(entradas, peso1, peso2, bias):
    for entrada in entradas:
        soma = (entrada[0] * peso1) + (entrada[1] * peso2) + bias
        print(f'Entrada: {entrada} => Saída: {step_activation(soma)}')

entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
saidas_esperadas = [0, 1, 1, 0]

peso1, peso2, bias = treinar_neuronio(entradas, saidas_esperadas)

testar_neuronio(entradas, peso1, peso2, bias)
