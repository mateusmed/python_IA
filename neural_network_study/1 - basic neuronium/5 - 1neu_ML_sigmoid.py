"""
Este script treina um neurônio usando a função de ativação sigmoide para aprender a função XOR.
Ele aplica o gradiente descendente para ajustar os pesos e o bias, e o treinamento
continua até que a taxa de acerto desejada seja atingida ou o número máximo de épocas seja alcançado.
Após o treinamento, o neurônio é testado.
"""


import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def inicializar_pesos():
    peso1 = random.gauss(0, 1)
    peso2 = random.gauss(0, 1)
    bias = random.gauss(0, 1)
    return peso1, peso2, bias

def treinar_neuronio(entradas, saidas_esperadas, taxa_acerto=95, epocas_max=100000):
    peso1, peso2, bias = inicializar_pesos()
    taxa_aprendizado = 0.1
    epocas = 0
    while epocas < epocas_max:
        erro_total = 0
        acertos = 0
        for i in range(len(entradas)):
            soma = (entradas[i][0] * peso1) + (entradas[i][1] * peso2) + bias
            saida = sigmoid(soma)
            erro = saidas_esperadas[i] - saida
            peso1 += taxa_aprendizado * erro * sigmoid_derivative(saida) * entradas[i][0]
            peso2 += taxa_aprendizado * erro * sigmoid_derivative(saida) * entradas[i][1]
            bias += taxa_aprendizado * erro * sigmoid_derivative(saida)
            erro_total += erro ** 2
            if abs(erro) < 0.5:
                acertos += 1

        taxa_acerto_atual = (acertos / len(entradas)) * 100
        taxa_falha = 100 - taxa_acerto_atual

        if epocas % 1000 == 0:
            print(f'Época {epocas}: Taxa de acerto = {taxa_acerto_atual:.2f}%, Falhas = {taxa_falha:.2f}%, Taxa de aprendizado = {taxa_aprendizado:.4f}')

        if taxa_acerto_atual >= taxa_acerto:
            print(f'Treinamento finalizado após {epocas+1} épocas, taxa de acerto alcançada: {taxa_acerto_atual:.2f}%')
            break

        epocas += 1

    return peso1, peso2, bias, taxa_acerto_atual, taxa_falha, epocas

def testar_neuronio(entradas, peso1, peso2, bias):
    print("Testando a rede treinada...")
    for entrada in entradas:
        soma = (entrada[0] * peso1) + (entrada[1] * peso2) + bias
        print(f'Entrada: {entrada} => Saída: {sigmoid(soma):.2f}')

entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
saidas_esperadas = [0, 1, 1, 0]
epocas_max = 50000

peso1, peso2, bias, taxa_acerto_atual, taxa_falha, epocas = treinar_neuronio(entradas, saidas_esperadas, taxa_acerto=95, epocas_max=epocas_max)

print(f'\nPesos finais: peso1 = {peso1:.4f}, peso2 = {peso2:.4f}, bias = {bias:.4f}')
print(f'Taxa de acerto final: {taxa_acerto_atual:.2f}%')
print(f'Taxa de falha final: {taxa_falha:.2f}%')

testar_neuronio(entradas, peso1, peso2, bias)
