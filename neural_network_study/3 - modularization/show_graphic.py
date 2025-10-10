import matplotlib.pyplot as plt
import numpy as np

try:
    from neuralnet import forward_pass
except ImportError:
    print("Aviso: Não foi possível importar 'forward_pass' de 'neuralnet'.")
    def forward_pass(network, point):
        print("Erro: A função 'forward_pass' não está disponível!")
        return [0]


def plot_errors_per_epoch(erros_por_epoca):
    """Plota o erro médio por época de treinamento e salva em arquivo."""
    fig = plt.figure(figsize=(10, 5))
    plt.plot(erros_por_epoca)
    plt.title("Erro Médio por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Erro (MSE)")
    plt.grid(True)
    
    filename = "plot_1_errors.png"
    plt.savefig(filename)
    print(f"Gráfico de erros salvo como '{filename}'")
    plt.close(fig)


def plot_output_surface(network, X_train, y_train):
    """Plota a superfície de saída da rede neural em 3D e salva em arquivo."""
    if X_train.shape[1] != 2:
        print("Esta função é para visualizar a saída de dados de entrada 2D.")
        return

    x_range = np.arange(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 0.1)
    y_range = np.arange(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 0.1)
    xx, yy = np.meshgrid(x_range, y_range)

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = [forward_pass(network, point)[0] for point in grid_points]
    Z = np.array(predictions).reshape(xx.shape)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c=y_train, marker='o', s=100, depthshade=True, edgecolor='k', cmap=plt.cm.RdBu)

    ax.set_title("Superfície de Saída da Rede Neural")
    ax.set_xlabel("Entrada X1")
    ax.set_ylabel("Entrada X2")
    ax.set_zlabel("Saída da Rede (Z)")
    ax.set_zlim(-0.1, 1.1)
    
    filename = "plot_2_surface.png"
    plt.savefig(filename)
    print(f"Gráfico de superfície 3D salvo como '{filename}'")
    plt.close(fig)


def plot_decision_boundary(network, X, y):
    """Plota a fronteira de decisão 2D de uma rede neural e salva em arquivo."""
    if X.shape[1] != 2:
        print("A visualização da fronteira de decisão só é suportada para dados de entrada 2D.")
        return

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = [round(forward_pass(network, point)[0]) for point in grid_points]
    Z = np.array(predictions).reshape(xx.shape)

    fig = plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolor='k', s=40)
    plt.title("Fronteira de Decisão da Rede Neural")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    handles, labels = scatter.legend_elements()
    plt.legend(handles, labels, title="Classes")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = "plot_3_boundary.png"
    plt.savefig(filename)
    print(f"Gráfico de fronteira de decisão salvo como '{filename}'")
    plt.close(fig)