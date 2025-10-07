
import numpy as np
import matplotlib.pyplot as plt
import activation_functions as activation_functions


def one_neuronium_simple(inputs,
                         pesos,
                         bias,
                         activation_func):

    soma = np.dot(inputs, pesos) + bias
    return activation_func(soma)


def create_neuron(neuron_id,
                  num_inputs,
                  activation_function = activation_functions.step_activation):
    neuron = {
        "id": neuron_id,
        "weights": np.random.randn(num_inputs),
        "bias": np.random.randn(),
        "activation": activation_function,
        "output": 0.0,
        "delta": 0.0,
        "log": []
    }
    return neuron


def create_layer(num_neurons,
                 num_inputs_per_neuron,
                 layer_number):
    layer = []
    for neuron_index in range(1, num_neurons + 1):
        neuron_id = f"{layer_number}.{neuron_index}"
        layer.append(create_neuron(neuron_id, num_inputs_per_neuron))

    return layer



"""
# começa em 1! os dados de entrada nao são considerados neuronios
mas eles são propagadaos para ambos os neuronios internos

entrada1      n1
                        saida
entrada2      n2

entrada1 -> n1
entrada1 -> n2
entrada2 -> n1
entrada2 -> n2
n1 -> saida
n2 -> saida
"""
def build_network(layer_structure, activation_func):

    network = []

    for i in range(1, len(layer_structure)):
        num_inputs = layer_structure[i-1]   # número de entradas da camada anterior
        num_neurons = layer_structure[i]    # número de neurônios nesta camada

        layer = create_layer(num_neurons, num_inputs, i)
        network.append(layer)

    return network


def forward_pass(network, input_vector):

    inputs = input_vector
    for layer_index, layer in enumerate(network):
        layer_outputs = []
        for neuron in layer:

            calc = np.dot(neuron["weights"], inputs) + neuron["bias"]
            activation_response = neuron["activation"](calc)
            neuron["output"] = activation_response
            layer_outputs.append(neuron["output"])

            neuron["log"].append({
                "inputs": inputs.copy(),
                "weighted_sum": calc,
                "output": neuron["output"]
            })
        inputs = layer_outputs
    return inputs


"""
    Calcula o delta de cada neurônio e atualiza pesos e bias.
"""
def backward_pass(network, input_vector, expected_output, learning_rate):

    for layer_index in reversed(range(len(network))):
        layer = network[layer_index]
        errors = []

        if layer_index == len(network) - 1:
            for neuron_index, neuron in enumerate(layer):
                if np.isscalar(expected_output):
                    error = neuron["output"] - expected_output
                else:
                    error = neuron["output"] - expected_output[neuron_index]
                errors.append(error)
        else:
            for neuron_index, neuron in enumerate(layer):
                error = 0.0
                next_layer = network[layer_index + 1]
                for next_neuron in next_layer:
                    error += next_neuron["weights"][neuron_index] * next_neuron["delta"]
                errors.append(error)

        for neuron_index, neuron in enumerate(layer):
            output = neuron["output"]
            neuron["delta"] = errors[neuron_index] * activation_functions.sigmoid_derivative(output)

            if layer_index == 0:
                inputs_to_use = input_vector
            else:
                inputs_to_use = [n["output"] for n in network[layer_index - 1]]
            for i in range(len(neuron["weights"])):
                neuron["weights"][i] -= learning_rate * neuron["delta"] * inputs_to_use[i]

            neuron["bias"] -= learning_rate * neuron["delta"]






def train_network(network, input_value, output_value, epochs, learning_rate=0.1):
    """
    Treina uma rede neural simples com forward e backward pass.

    network: lista de camadas (cada camada é uma lista de neurônios)
    X: entradas (matriz de treino)
    y: saídas esperadas (vetor)
    epochs: número de épocas de treino
    learning_rate: taxa de aprendizado
    """
    erros_por_epoca = []

    for epoch in range(epochs):
        erro_total = 0.0

        for i in range(len(input_value)):
            input_vector = input_value[i]
            expected_output = output_value[i]

            # ======= Forward pass =======
            forward_pass(network, input_vector)

            # ======= Backward pass =======
            backward_pass(network, input_vector, expected_output, learning_rate)

            # ======= Calcula erro (MSE parcial) =======
            output_layer = network[-1]
            outputs = np.array([neuron["output"] for neuron in output_layer])
            erro_total += np.mean((expected_output - outputs) ** 2)

        erro_medio = erro_total / len(input_value)
        erros_por_epoca.append(erro_medio)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Época {epoch+1}/{epochs} - Erro médio: {erro_medio:.6f}")

    # ======= Plot do erro =======
    plt.plot(erros_por_epoca)
    plt.title("Erro médio por época")
    plt.xlabel("Épocas")
    plt.ylabel("Erro (MSE)")
    plt.grid(True)
    plt.show()

    return network




"""
    Treina um perceptron simples (um único neurônio) e mostra progresso se verbose=True.

    x: entradas (array NxM)
    y: saídas esperadas (array Nx1)
    activation_function: função de ativação do neurônio
    learning_rate: taxa de aprendizado
    epochs: número de épocas de treino
    verbose: se True, imprime progresso
    """
def train_perceptron(x,
                     y,
                     activation_function,
                     learning_rate=0.1,
                     epochs=20,
                     verbose=True):

    n_features = x.shape[1]
    pesos = np.random.rand(n_features)
    bias = np.random.rand(1)

    for epoca in range(epochs):
        if verbose:
            print(f"\nÉpoca {epoca+1}/{epochs}")
        for i in range(len(x)):
            entrada = x[i]
            esperado = y[i]

            # saída do neurônio
            soma = np.dot(entrada, pesos) + bias
            obtido = activation_function(soma)

            # erro
            erro = esperado - obtido

            # atualização dos pesos e bias
            pesos += learning_rate * erro * entrada
            bias += learning_rate * erro

            if verbose:
                print(f"Entrada: {entrada}, Esperado: {esperado}, Obtido: {obtido}, "
                      f"Pesos: {pesos}, Bias: {bias}")

    return pesos, bias



