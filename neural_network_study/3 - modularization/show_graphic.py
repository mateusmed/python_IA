
import matplotlib.pyplot as plt






def plot(erros_por_epoca):
    plt.plot(erros_por_epoca)
    plt.title("Erro médio por época")
    plt.xlabel("Épocas")
    plt.ylabel("Erro (MSE)")
    plt.grid(True)
    plt.show()



#
# def plot_grafic():
#     # Intervalo de valores para plotar
#     x = np.linspace(-10, 10, 400)
#
#     # Saídas
#     y_sigmoid = sigmoid(x)
#     y_relu = relu(x)
#     y_step = step_activation(x)
#
#     # Criando subplots
#     fig, axs = plt.subplots(1, 3, figsize=(15, 4))
#
#     # Plot Sigmoid
#     axs[0].plot(x, y_sigmoid, 'b')
#     axs[0].set_title('Sigmoid')
#     axs[0].grid(True)
#
#     # Plot ReLU
#     axs[1].plot(x, y_relu, 'g')
#     axs[1].set_title('ReLU')
#     axs[1].grid(True)
#
#     # Plot Step
#     axs[2].plot(x, y_step, 'r')
#     axs[2].set_title('Step (Degrau)')
#     axs[2].grid(True)
#
#     plt.suptitle('Funções de Ativação', fontsize=14)
#     plt.show()