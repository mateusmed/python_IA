
Primeiros Conceitos (Década de 1940 e 1950):

Em 1943, Warren McCulloch e Walter Pitts criaram um modelo de "neurônio artificial", \
inspirado nos neurônios biológicos, para representar operações lógicas simples. \
O objetivo não era criar um único código para resolver qualquer problema, \
mas sim representar o funcionamento do cérebro de forma abstrata.

Perceptron (1958):

Em 1958, Frank Rosenblatt desenvolveu o Perceptron, um modelo de rede neural \
capaz de resolver problemas simples de classificação, como distinguir entre dois tipos de entradas. \
A ideia era que, ao invés de programar regras específicas, a rede aprenderia com exemplos, \
criando um sistema adaptável.



O motivo de usar redes neurais é realizar o processo de machine learning, \
onde você alimenta o sistema com muitos dados para que ele 'calibre' seus parâmetros \
(pesos) de modo a mapear as entradas para as saídas corretas. \
Esse processo geralmente é feito por meio de aprendizado supervisionado

O modelo (rede neural) é determinístico depois que está treinado.\
Ou seja, se você der a mesma entrada, vai sair sempre o mesmo resultado.

Durante o treinamento, pode haver variações \
(como pesos sendo inicializados aleatoriamente, otimizações estocásticas etc.). \
Mas uma vez treinada, a rede funciona como uma função f(x) fixa.


Uma das grandes forças das redes neurais é sua capacidade de modelar e resolver problemas linerares não lineares,
o que isso significa?

Problemas Lineares: A relação entre as variáveis é simples e pode ser representada por uma linha reta (ex: 
y=2x+3).

Problemas Não Lineares: A relação é mais complexa, e não pode ser representada por uma linha reta (ex:
y=x^2 +3x+5).


Ao estudarmos apenas um unico neuronio caimos no problema de "variavies não linearmente separadas" 

o que isso sinifica, significa que qualquer problema que você solicita a uma rede neural,
ela vai transformar isso em um problema numerico e tentar traçar "linhas" que se aproximam da
solução que você propôs, por exemplo
