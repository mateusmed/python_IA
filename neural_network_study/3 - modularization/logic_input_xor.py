
import numpy as np
import activation_functions as activation_functions
import neuralnet as neuralnet
import serialization as serialization



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

    network = neuralnet.build_network([2, 2, 1])

    neuralnet.train_network(network, input_logic_ports, port_and_result_expected, 1000)

    trained_net, historico_erros = neuralnet.train_network( network,
                                                            input_value=input_logic_ports,
                                                            output_value=port_and_result_expected,
                                                            epochs=10000 )

    # 3. Salva a rede treinada
    serialization.save_network_state(trained_net, filename="xor_trained_model.pkl")

    # print("Rede inicial (pesos e bias aleatórios):")
    # for layer in network:
    #     for neuron in layer:
    #         print(neuron)

    # Testa a rede
    print("\nResultados com pesos aleatórios (não resolve ainda o XOR):")
    for input_value, expected in zip(input_logic_ports, port_and_result_expected):
        saida = neuralnet.forward_pass(network, input_value)
        print(f"Entrada: {input_value} | Esperado: {expected} | Obtido: {saida}")


if __name__ == "__main__":
    main()


