

import matplotlib.pyplot as plt
import numpy as np
import network as network
import activation_functions as activation_functions


def execute_input(input_matrix, pesos, bias):
    outputs = []
    for input in input_matrix:
        saida = network.one_neuronium_simple(input,
                                             pesos,
                                             bias,
                                             activation_functions.step_activation)
        outputs.append(saida)

    return outputs


def show_result(input_matrix, output_expected, real_outputs):
    print("\nResultados finais:")
    for input, expected, result in zip(input_matrix, output_expected, real_outputs):
        print(f"Entrada: {input} | Esperado: {expected} | Obtido: {result}")


def plot_visual():
    fig, ax = plt.subplots()
    for i, entrada in enumerate(input_logic_ports):
        cor = 'blue' if outputs[i] == 1 else 'red'
        ax.scatter(entrada[0], entrada[1], c=cor, s=100,
                   label=f"out={outputs[i]}" if i < 2 else "")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Perceptron treinado")
    ax.legend()
    ax.grid(True)
    plt.show()



def main():

    input_logic_ports = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    port_and_result_expected = np.array([
        0,
        0,
        0,
        1
    ])

    pesos, bias = network.train_perceptron(input_logic_ports,
                                           port_and_result_expected,
                                           activation_functions.step_activation,
                                           learning_rate=0.1,
                                           epochs=20)

    print("Pesos treinados:", pesos)
    print("Bias treinado:", bias)

    output = execute_input(input_logic_ports, pesos, bias)
    show_result(input_logic_ports, port_and_result_expected, output)


main()