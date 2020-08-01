#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 09:17:48 2019

@author: mateus.santos
"""

from math import sqrt

# o numero poderia ser o numero de download
# dai calcular a distancia euclidiana

# faria mais sentido dado o usuário similar, qual jogo que ele mais usa;
# o numero poderia ser o numero de uso
# dai calcular a distancia euclidiana e a distancia de manhattan

'''
{'Ana':
{'Freddy x Jason': <numero_de_uso>,
'O Ultimato Bourne': <numero_de_uso>,
'Star Trek': <numero_de_uso>,
'Exterminador do Futuro': <numero_de_uso>,
'Norbit': <numero_de_uso>,
'Star Wars': <numero_de_uso>},
'''


'''
    if len(item_a) < len(item_b):
        item_temp = item_a
        item_a = item_b
        item_b = item_temp    
'''


def get_relation(item_a, item_b):
    relation = {}
    relation['commun'] = []
    relation['not_commun'] = []

    if len(item_a) < len(item_b):
        item_temp = item_a
        item_a = item_b
        item_b = item_temp

    for item in item_a:
        if item in item_b:
            relation['commun'].append(item)
        else:
            relation['not_commun'].append(item)

    return relation


# dont use
def has_commum(relation):
    if len(relation['commun']) == 0:
        return False
    return True


def similarity_by_strategy(item_a, item_b, strategy_score):
    relation = get_relation(item_a, item_b)
    relation['score'] = strategy_score(item_a, item_b, relation)


def comum_by_total_strategy(relation):
    total = len(relation['commun']) + len(relation['not_commun'])
    return len(relation['commun']) / total


# similaridade por comparação de itens
def similarity(item_a, item_b):
    relation = get_relation(item_a, item_b)
    total = len(relation['commun']) + len(relation['not_commun'])
    relation['score'] = len(relation['commun']) / total
    return relation


# similaridade por distancia euclidiana
def euclidian(item_a, item_b):
    relation = get_relation(item_a, item_b)
    # se não tem relação em comum return
    soma = 0

    for commun in relation['commun']:
        if (item_a[commun] > item_b[commun]):
            soma = soma + pow((item_a[commun] - item_b[commun]), 2)
        else:
            soma = soma + pow((item_b[commun] - item_a[commun]), 2)

    if soma == 0:
        return 0

    return porcent(sqrt(soma))


# similaridade por distancia de manhatann
def manhattan(item_a, item_b):
    relation = get_relation(item_a, item_b)
    soma = 0

    for commun in relation['commun']:

        if (item_a[commun] > item_b[commun]):
            soma = soma + (item_a[commun] - item_b[commun])
        else:
            soma = soma + (item_b[commun] - item_a[commun])

    if soma == 0:
        return 0

    return porcent(soma)


def porcent(value):
    return 1 / (1 + value)


def all_similar(item_a, data):
    similarity_list = []

    for name in data:
        item_b = data[name]

        if item_b != item_a:
            obj = {}
            obj['relation'] = similarity(item_a, item_b)
            obj['item'] = name

            similarity_list.append(obj)

    similarity_list.sort(key=lambda k: k['relation']['score'], reverse=True)

    return similarity_list


# first find
def get_similar_item_by_score(item_a, data, score):
    for name in data:
        item_b = data[name]
        if item_b != item_a:

            relation = similarity(item_a, item_b)

            if (relation['score'] >= score):
                relation['item'] = name
                return relation


# ta funcionando maneiro não; refazer
# prediga a nota para os apks que eu não vi
def predict_similar_score(item_predict, all_similar, user_media):
    data_predict = {}
    data_predict['score_total'] = 0

    for item_similar in all_similar:

        name_item = item_similar['item']
        print('------>', name_item)
        score = item_similar['relation']['score']

        for name_item_not_commun in item_similar['relation']['not_commun']:

            print('item is not commun --->', name_item_not_commun)

            # if o item incomum não pertence no meu item que pretendo fazer a predição
            # ou seja, se ele não baixou o o apk
            if name_item_not_commun not in item_predict:

                result = user_media[name_item][name_item_not_commun]
                total = (score * result)

                if name_item_not_commun not in data_predict:
                    data_predict[name_item_not_commun] = 0
                else:
                    data_predict[name_item_not_commun] = data_predict[name_item_not_commun] + total

    return data_predict
