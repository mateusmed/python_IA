#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_service import user_media, media_tag
import logic


def recomendation_by_user():
    ana = user_media['Ana']
    pedro = user_media['Pedro']
    claudia = user_media['Claudia']
    marcos = user_media['Marcos']
    mateus = user_media['Mateus']
    gabi = user_media['Gabi']

    # tudo que tem em pedro tem na ana
    logic.get_relation(pedro, ana)

    value = ana['Freddy x Jason']
    print(value)

    logic.similarity(pedro, claudia)

    logic.euclidian(pedro, claudia)
    logic.manhattan(mateus, gabi)

    logic.similarity(ana, mateus)

    all_similares = logic.all_similar(mateus, user_media)
    print(all_similares)

    logic.predict_similar_score(pedro, all_similares, user_media)

    similar_user = logic.get_similar_item_by_score(mateus, user_media, 0.8)
    print(similar_user['not_commun'])


def recomendation_by_tag():
    media_a = media_tag['Star Trek']
    media_b = media_tag['O Ultimato Bourne']
    media_c = media_tag['Exterminador do Futuro']
    media_d = media_tag['Freddy x Jason']

    relation = logic.get_relation(media_a, media_c)
    # print(relation)

    similarity = logic.similarity(media_a, media_b)
    # print(similarity)

    allsimilarity = logic.all_similar(media_d, media_tag)
    # print('all similarity', allsimilarity)

    tag = logic.get_similar_item_by_score(media_a, media_tag, 0.6)
    print(tag)


def verify_similarity():
    starTrek = media_tag['Star Trek']
    ultimatoBourne = media_tag['O Ultimato Bourne']
    starWars = media_tag['Star Wars']

    # relation = logic.get_relation(starTrek, ultimatoBourne)
    # print("relation: ", relation)
    #
    # similarity = logic.similarity(starTrek, ultimatoBourne)
    # print("similarity: ", similarity)

    all_similar = logic.all_similar(starTrek, media_tag)
    print("all_similar: {}".format(all_similar[0]))

    print("--------------------")

    # relation = logic.get_relation(starTrek, starWars)
    # print("relation: ", relation)
    #
    # similarity = logic.similarity(starTrek, starWars)
    # print("similarity: ", similarity)
    #
    # all_similar = logic.all_similar(media_a, media_tag)
    # print("all_similar:", all_similar)


if __name__ == "__main__":
    verify_similarity()