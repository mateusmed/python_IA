
import numpy as np
import activation_functions as activation_functions
import neuralnet as neuralnet



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

    network = neuralnet.build_network([2, 2, 1],
                                      activation_functions.step_activation)

    neuralnet.train_network(network, input_logic_ports, port_and_result_expected, 1000)

    print("Rede inicial (pesos e bias aleatórios):")
    for camada in network:
        for neuron in camada:
            print(neuron)

    # Testa a rede
    print("\nResultados com pesos aleatórios (não resolve ainda o XOR):")
    for entrada, esperado in zip(input_logic_ports, port_and_result_expected):
        saida = neuralnet.backward_pass(network, entrada, port_and_result_expected)
        print(f"Entrada: {entrada} | Esperado: {esperado} | Obtido: {saida}")


if __name__ == "__main__":
    main()


