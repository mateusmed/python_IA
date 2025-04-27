
"""
Este script treina um neurônio para aprender a função XOR usando o algoritmo de gradiente descendente,
ajustando pesos e bias com base nos erros. O treinamento continua até atingir uma
taxa de acerto desejada ou o número máximo de épocas. Após o treinamento, o neurônio é testado.
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

def treinar_neuronio(entradas, saidas_esperadas, taxa_aprendizado=0.1, epocas=10000, taxa_acerto=95):
    peso1, peso2, bias = inicializar_pesos()

    for epoca in range(epocas):
        erro_total = 0
        acertos = 0
        for i in range(len(entradas)):
            soma = (entradas[i][0] * peso1) + (entradas[i][1] * peso2) + bias
            saida = step_activation(soma)

            erro = saidas_esperadas[i] - saida

            peso1 += taxa_aprendizado * erro * entradas[i][0]
            peso2 += taxa_aprendizado * erro * entradas[i][1]
            bias += taxa_aprendizado * erro

            erro_total += erro ** 2

            if erro == 0:
                acertos += 1

        taxa_acerto_atual = (acertos / len(entradas)) * 100
        taxa_falha = 100 - taxa_acerto_atual

        if taxa_acerto_atual >= taxa_acerto:
            print(f'Treinamento finalizado após {epoca+1} épocas, taxa de acerto alcançada: {taxa_acerto_atual:.2f}%')
            break

        if epoca % 1000 == 0:
            print(f'Época {epoca}: Taxa de acerto = {taxa_acerto_atual:.2f}%, Falhas = {taxa_falha:.2f}%')

    return peso1, peso2, bias, taxa_acerto_atual, taxa_falha

def testar_neuronio(entradas, peso1, peso2, bias):
    print("Testando a rede treinada...")
    for entrada in entradas:
        soma = (entrada[0] * peso1) + (entrada[1] * peso2) + bias
        print(f'Entrada: {entrada} => Saída: {step_activation(soma)}')

entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
saidas_esperadas = [0, 1, 1, 0]

epocas = 50000
taxa_aprendizado = 0.05
taxa_acerto = 95

peso1, peso2, bias, taxa_acerto_atual, taxa_falha = treinar_neuronio(entradas, saidas_esperadas, taxa_aprendizado, epocas, taxa_acerto)

print(f'\nPesos finais: peso1 = {peso1:.4f}, peso2 = {peso2:.4f}, bias = {bias:.4f}')
print(f'Taxa de acerto final: {taxa_acerto_atual:.2f}%')
print(f'Taxa de falha final: {taxa_falha:.2f}%')

testar_neuronio(entradas, peso1, peso2, bias)
