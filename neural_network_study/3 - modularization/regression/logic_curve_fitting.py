import numpy as np
import neuralnet as neuralnet
import serialization as serialization
import show_graphic as show_graphic
import activation_functions as activation_functions

def main():
    # --- 1. Geração de Dados para Regressão ---
    # Gera 50 pontos de dados para a função seno com algum ruído
    np.random.seed(42) # Para resultados reproduzíveis
    X = np.linspace(-np.pi, np.pi, 50).reshape(50, 1)
    y = np.sin(X) + np.random.normal(0, 0.15, (50, 1))

    # --- 2. Construção da Rede para Regressão ---
    # Estrutura: 1 entrada -> 2 camadas ocultas com 32 neurônios -> 1 saída
    network = neuralnet.build_network([1, 16, 16, 1])

    # Altera a função de ativação da camada de SAÍDA para linear
    # Isso é crucial para problemas de regressão, para não limitar a saída entre 0 e 1
    output_layer = network[-1]
    for neuron in output_layer:
        neuron["activation_func"] = activation_functions.linear
        neuron["activation_deriv"] = activation_functions.linear_derivative

    # --- 3. Treinamento da Rede ---
    print("Iniciando treinamento da rede para o problema de Curve Fitting...")
    trained_net, historico_erros = neuralnet.train_network(
        network=network,
        input_value=X,
        output_value=y,
        epochs=5000,       # Reduzido de 20000
        learning_rate=0.01  # Aumentado de 0.001
    )
    print("Treinamento concluído.")

    # --- 4. Salvar e Testar a Rede (Opcional) ---
    serialization.save_network_state(trained_net, filename="curve_fitting_model")

    # --- 5. Visualização dos Resultados ---
    print("\nGerando gráficos de visualização...")
    
    # Gráfico 1: Erro por Época
    show_graphic.plot_errors_per_epoch(historico_erros)

    # Gráfico 2: Curve Fit
    show_graphic.plot_curve_fit(trained_net, X, y)

    # Gráfico 3: Arquitetura da Rede
    show_graphic.plot_network_architecture(network=trained_net)

if __name__ == "__main__":
    main()
