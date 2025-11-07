import numpy as np
import activation_functions as activation_functions


def create_neuron(neuron_id,
                  num_inputs,
                  activation_func = activation_functions.sigmoid,
                  activation_deriv = activation_functions.sigmoid_derivative): 
    neuron = {
        "id": neuron_id,
        "weights": np.random.randn(num_inputs),
        "bias": np.random.randn(),
        "activation_func": activation_func,      # Função de ativação (usada no forward pass)
        "activation_deriv": activation_deriv,    # Derivada da função de ativação (usada no backward pass)
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

        neuron = create_neuron(neuron_id, num_inputs_per_neuron)
        layer.append(neuron)

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
def build_network(layer_structure):

    network = []

    for i in range(1, len(layer_structure)):
        num_inputs = layer_structure[i-1]   # número de entradas da camada anterior
        num_neurons = layer_structure[i]    # número de neurônios nesta camada

        layer = create_layer(num_neurons, num_inputs, i)
        network.append(layer)

    return network

## PEQUENO AJUSTE EM FORWARD PASS
def forward_pass(network, input_vector, return_all_layers=False):
    inputs = input_vector
    all_activations = [inputs]  # guarda as ativações de cada camada

    for layer_index, layer in enumerate(network):
        layer_outputs = []
        for neuron in layer:
            calc = np.dot(neuron["weights"], inputs) + neuron["bias"]
            activation_response = neuron["activation_func"](calc)
            neuron["output"] = activation_response
            layer_outputs.append(neuron["output"])
        inputs = np.array(layer_outputs)
        all_activations.append(inputs)

    if return_all_layers:
        return all_activations  # lista: [entrada, camada1, camada2, ...]
    else:
        return inputs  # só a saída final


def calculate_layer_errors(network, layer_index, expected_output):
    layer = network[layer_index]
    errors = []

    if layer_index == len(network) - 1:
        # Camada de saída
        for neuron_index, neuron in enumerate(layer):
            if np.isscalar(expected_output):
                error = neuron["output"] - expected_output
            else:
                error = neuron["output"] - expected_output[neuron_index]
            errors.append(error)
    else:
        # Camada oculta
        next_layer = network[layer_index + 1]
        for neuron_index, neuron in enumerate(layer):
            error = 0.0
            for next_neuron in next_layer:
                error += next_neuron["weights"][neuron_index] * next_neuron["delta"]
            errors.append(error)
    return errors



"""
    Calcula o delta do neurônio (erro * derivada da ativação).
"""
def calculate_neuron_delta(neuron, error):
    output = neuron["output"]
    # Usando a derivada armazenada no neurônio para consistência
    neuron["delta"] = error * neuron["activation_deriv"](output) 
    return neuron["delta"]

"""
    Obtém o vetor de entrada apropriado para uma camada.
"""
def get_inputs_for_layer(network, layer_index, input_vector):

    if layer_index == 0:
        return input_vector
    else:
        inputs = []
        for neuron in network[layer_index - 1]:
            inputs.append(neuron["output"])
        return np.array(inputs)

"""
    Atualiza os pesos e bias do neurônio.
"""
def update_neuron_weights(neuron, inputs, learning_rate):

    for i in range(len(neuron["weights"])):
        new_weights = (learning_rate * neuron["delta"] * inputs[i])
        neuron["weights"][i] = neuron["weights"][i] - new_weights

    neuron["bias"] = neuron["bias"] - (learning_rate * neuron["delta"])


"""Executa o backpropagation camada por camada."""
def backward_pass(network, input_vector, expected_output, learning_rate):

    for layer_index in reversed(range(len(network))):
        layer = network[layer_index]
        errors = calculate_layer_errors(network, layer_index, expected_output)

        for neuron_index, neuron in enumerate(layer):
            calculate_neuron_delta(neuron, errors[neuron_index])
            inputs_to_use = get_inputs_for_layer(network, layer_index, input_vector)
            update_neuron_weights(neuron, inputs_to_use, learning_rate)


"""
    Processa um vetor de entrada através da rede treinada 
    e retorna a saída bruta da última camada.
"""
def predict(network, input_vector):

    raw_output = forward_pass(network, input_vector)

    # # Para classificação (como XOR), transformamos a saída bruta em 0 ou 1
    # # O limite 0.5 é comum para a Sigmoide.
    # classification = (raw_output >= 0.5).astype(int)

    # Retorna o valor bruto e o valor classificado
    # classification
    return raw_output,


"""
    Treina uma rede neural simples com forward e backward pass.

    network: lista de camadas (cada camada é uma lista de neurônios)
    X: entradas (matriz de treino)
    y: saídas esperadas (vetor)
    epochs: número de épocas de treino
    learning_rate: taxa de aprendizado
"""
def train_network(network, input_value, output_value, epochs, learning_rate=0.1):

    erros_por_epoca = []

    for epoch in range(epochs):
        erro_total = 0.0

        for i in range(len(input_value)):
            input_vector = input_value[i]
            expected_output = output_value[i]

            # ======= Forward pass =======
            # Usamos o forward_pass no modo de treino (com log e atualização de output)
            forward_pass(network, input_vector)

            # ======= Backward pass =======
            backward_pass(network, input_vector, expected_output, learning_rate)

            # ======= Calcula erro (MSE parcial) =======
            output_layer = network[-1]
            outputs = np.array([neuron["output"] for neuron in output_layer])
            erro_total += np.mean((expected_output - outputs) ** 2)

        erro_medio = erro_total / len(input_value)
        erros_por_epoca.append(erro_medio)

        if epoch % 1000 == 0 or epoch == epochs - 1: # Ajustei a frequência de print para épocas longas
            print(f"Época {epoch+1}/{epochs} - Erro médio: {erro_medio:.6f}")


    # Retorna a rede (com os pesos ajustados) e o histórico de erros
    return network, erros_por_epoca

