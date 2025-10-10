
import pickle
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    """
    Classe de codificador JSON especial para lidar com tipos de dados NumPy.
    Converte arrays NumPy em listas e tipos numéricos NumPy em tipos Python nativos.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic, np.number)):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)


"""
    [PRIVADO] Salva apenas os pesos e bias da rede em formato legível (JSON).
    Chamado internamente por save_network_state.
    """

def save_network_state_json(network, filename):
    network_data = []

    for layer in network:
        layer_data = []
        for neuron in layer:
            # Converte os arrays NumPy para listas Python para o JSON

            print(f"{neuron}")

            neuron_data = {
                "id": neuron["id"],
                "weights": neuron["weights"],
                "bias": neuron["bias"],
                # "log": neuron["log"]
            }
            layer_data.append(neuron_data)
        network_data.append(layer_data)

    try:
        with open(filename, 'w') as file:
            # Usa 'indent=4' para formatar o JSON de forma legível por humanos
            json.dump(network_data, file, indent=4, cls=NumpyEncoder)
        print(f"Estado legível da rede salvo com sucesso em '{filename}' (JSON)")
    except Exception as e:
        print(f"Erro ao salvar a rede em JSON: {e}")


"""
    Salva o estado completo da rede (pesos, bias e referências de função)
    usando a biblioteca pickle.
"""
def save_network_state(network, filename="network_state", create_new_file_legible=True):

    extension = ".pkl"
    filename = filename + extension

    try:
        with open(filename, 'wb') as file:
            pickle.dump(network, file)
        print(f"\nEstado da rede salvo com sucesso em '{filename}'")
    except Exception as e:
        print(f"\nErro ao salvar a rede: {e}")

    if create_new_file_legible:

        json_filename = filename.replace('.pkl', '.json')
        save_network_state_json(network, json_filename)

"""
Carrega o estado da rede de um arquivo pickle.
"""
def load_network_state(filename="network_state.pkl"):
    try:
        with open(filename, 'rb') as file:
            network = pickle.load(file)
        print(f"\nEstado da rede carregado com sucesso de '{filename}'")
        return network
    except FileNotFoundError:
        print(f"\nErro: Arquivo '{filename}' não encontrado.")
        return None
    except Exception as e:
        print(f"\nErro ao carregar a rede: {e}")
        return None