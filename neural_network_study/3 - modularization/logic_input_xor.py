import numpy as np
import neuralnet as neuralnet
import serialization as serialization
import show_graphic as show_graphic

def main():

    input_logic_ports = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    port_and_result_expected = np.array([
        0,
        1,
        1,
        0
    ])

    # Constrói a estrutura da rede
    network = neuralnet.build_network([2, 3, 1])

    # Treina a rede
    print("Iniciando treinamento da rede para o problema XOR...")
    trained_net, historico_erros = neuralnet.train_network(
        network=network,
        input_value=input_logic_ports,
        output_value=port_and_result_expected,
        epochs=10000,
        learning_rate=0.1
    )
    print("Treinamento concluído.")

    # Salva a rede treinada (a função adiciona a extensão .pkl e .json)
    serialization.save_network_state(trained_net, filename="xor_trained_model")

    # Testa a rede treinada
    print("\nResultados com a rede treinada:")
    for input_value, expected in zip(input_logic_ports, port_and_result_expected):
        saida = neuralnet.forward_pass(trained_net, input_value)
        print(f"Entrada: {input_value} | Esperado: {expected} | Obtido: {round(saida[0], 4)}")

    print("\nGerando gráficos de visualização (em janelas separadas)...")
    # Gráfico 1: Erro por Época
    print("Mostrando Gráfico 1: Erro por Época")
    show_graphic.plot_errors_per_epoch(historico_erros)
    
    # Gráfico 2: Fronteira de Decisão 2D
    print("Mostrando Gráfico 2: Fronteira de Decisão 2D")
    show_graphic.plot_decision_boundary(
        network=trained_net,
        X=input_logic_ports,
        y=port_and_result_expected
    )

    # Gráfico 3: Superfície de Saída 3D
    print("Mostrando Gráfico 3: Superfície de Saída 3D")
    show_graphic.plot_output_surface(
        network=trained_net,
        X_train=input_logic_ports,
        y_train=port_and_result_expected
    )

if __name__ == "__main__":
    main()