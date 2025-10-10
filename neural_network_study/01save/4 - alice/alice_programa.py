


def imprimir_nome(nome):

    print(nome)

def somar_numeros(caixa1, caixa2):
    caixa3 = caixa1 + caixa2
    print(caixa3)
    return caixa3

def bts(jimin, jungkook, namjoon):
    todynho= jimin + jungkook + namjoon
    print(todynho)


def comprar_pao(dinheiro):

    if dinheiro == 0:
        print("dinheiro insuficiente")
        return dinheiro
    else:
        troco = dinheiro - 1
        print("comprei 1 real de pao")
        return troco

def executando_bts():
    print('executando função principal')
    paçoca = 80
    amendoim = 20
    mingau = 40

    bts(paçoca, mingau, amendoim)

def logica_substituicao():

    a = 10
    b = 20

    print("valor de A: ", a)
    print("valor de B: ", b)

    a=20
    b=10

    print("valor de A:", a)
    print("valor de B: ",b)

    c=a
    a=b

    b=c

    print("valor de A:", a)
    print("valor de B: ", b)



def mudar_nome(nome):
    nome = nome.replace("a", "X")
    print(nome)

def imprimir_loop(nome):

    for i in range(200):
        print(i , nome)


def main():

    nome = "alice"
    imprimir_loop(nome)

main()



