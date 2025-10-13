import numpy as np
import neuralnet as neuralnet
import serialization as serialization
import show_graphic as show_graphic
import activation_functions as activation_functions

def main():
    # --- 1. Geração de Dados para Regressão ---
    # Gera 50 pontos de dados simulando temperatura ao longo de um dia (seno com ruído)
    np.random.seed(42)
    X = np.linspace(0, 24, 50).reshape(50, 1)  # Horas do dia
    y = 10 + 10 * np.sin((X - 6) * np.pi / 12) + np.random.normal(0, 1.5, (50, 1))
    # A curva oscila entre ~0°C e ~20°C

    # --- 2. Construção da Rede ---
    network = neuralnet.build_network([1, 16, 16, 1])

    # Define função de ativação linear na saída
    output_layer = network[-1]
    for neuron in output_layer:
        neuron["activation_func"] = activation_functions.linear
        neuron["activation_deriv"] = activation_functions.linear_derivative

    # --- 3. Treinamento ---
    print("Iniciando treinamento da rede para previsão de temperatura...")
    trained_net, historico_erros = neuralnet.train_network(
        network=network,
        input_value=X,
        output_value=y,
        epochs=5000,
        learning_rate=0.01
    )
    print("Treinamento concluído.")

    # --- 4. Salvamento (opcional) ---
    serialization.save_network_state(trained_net, filename="temp_prediction_model")

    # --- 5. Visualização ---
    print("\nGerando gráficos de visualização...")
    show_graphic.plot_errors_per_epoch(historico_erros)
    show_graphic.plot_curve_fit(trained_net, X, y)
    show_graphic.plot_network_architecture(network=trained_net)

    # --- 6. Previsão de novos valores ---
    print("\nTestando previsões da rede treinada:")
    horas_para_prever = np.array([[6], [12], [18], [23]])  # Horários específicos
    for hora in horas_para_prever:
        saida = neuralnet.forward_pass(trained_net, hora)
        print(f"Hora: {hora[0]:>2}h | Temperatura prevista: {saida[0]:.2f} °C")

if __name__ == "__main__":
    main()
