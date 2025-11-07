import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

# --- Diretório para salvar os gráficos ---
OUTPUT_DIR = "plot"

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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, "plot_1_errors.png")
    plt.savefig(filename)
    print(f"Gráfico de erros salvo como '{filename}'")
    plt.close(fig)

def plot_output_surface(network, X_train, y_train):
    """Plota a superfície de saída da rede neural em 3D de vários ângulos e salva em arquivos."""
    if X_train.shape[1] != 2:
        print("Esta função é para visualizar a saída de dados de entrada 2D.")
        return

    x_range = np.arange(X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5, 0.1)
    y_range = np.arange(X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5, 0.1)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = [forward_pass(network, point)[0] for point in grid_points]
    Z = np.array(predictions).reshape(xx.shape)

    angles = [-60, 0, 60, 120]
    point_colors = ['red' if val == 0 else 'white' for val in y_train]

    for i, angle in enumerate(angles):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.8)
        ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c=point_colors, marker='o', s=100, depthshade=True, edgecolor='black', linewidths=1.5)

        ax.set_title("Superfície de Saída da Rede Neural")
        ax.set_xlabel("Entrada X1")
        ax.set_ylabel("Entrada X2")
        ax.set_zlabel("Saída da Rede (Z)")
        ax.set_zlim(-0.1, 1.1)
        
        ax.view_init(elev=30, azim=angle)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, f"plot_2_surface_angle_{i+1}.png")
        plt.savefig(filename)
        print(f"Gráfico de superfície 3D (ângulo {angle}°) salvo como '{filename}'")
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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, "plot_3_boundary.png")
    plt.savefig(filename)
    print(f"Gráfico de fronteira de decisão salvo como '{filename}'")
    plt.close(fig)

def plot_network_architecture(network):
    """Desenha a arquitetura da rede neural e salva em arquivo."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_title('Arquitetura da Rede Neural', fontsize=16)

    layer_sizes = [len(layer) for layer in network]
    num_layers = len(layer_sizes)
    
    x_start, x_end = 0.1, 0.9
    x_spacing = (x_end - x_start) / (num_layers - 1) if num_layers > 1 else 0
    
    neuron_positions = []
    for i, layer_size in enumerate(layer_sizes):
        layer_positions = []
        y_start, y_end = 0.1, 0.9
        y_spacing = (y_end - y_start) / (layer_size - 1) if layer_size > 1 else 0.5
        
        for j in range(layer_size):
            x = x_start + i * x_spacing
            y = y_start + j * y_spacing if layer_size > 1 else 0.5
            layer_positions.append((x, y))
        neuron_positions.append(layer_positions)

    for i in range(num_layers - 1):
        for source_pos in neuron_positions[i]:
            for target_pos in neuron_positions[i+1]:
                line = plt.Line2D((source_pos[0], target_pos[0]), (source_pos[1], target_pos[1]), color='gray', alpha=0.5, zorder=1)
                ax.add_line(line)

    radius = 0.03
    for i, layer_positions in enumerate(neuron_positions):
        for pos in layer_positions:
            if i == 0:
                color = 'skyblue'
            elif i == num_layers - 1:
                color = 'lightgreen'
            else:
                color = 'lightcoral'
            
            circle = plt.Circle(pos, radius, color=color, ec='black', zorder=2)
            ax.add_patch(circle)

    for i, layer_size in enumerate(layer_sizes):
        x = x_start + i * x_spacing
        ax.text(x, 1.0, f'Camada {i+1}\n({layer_size} neurônios)', ha='center', va='bottom', fontsize=10)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, "plot_4_architecture.png")
    plt.savefig(filename)
    print(f"Gráfico de arquitetura salvo como '{filename}'")
    plt.close(fig)

def plot_curve_fit(network, X_original, y_original):
    """Plota o resultado de um problema de regressão (curve fitting)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plota os dados originais com ruído
    ax.scatter(X_original, y_original, label='Dados Originais', color='blue', alpha=0.6)

    # Gera uma curva suave usando a rede treinada
    X_curve = np.linspace(X_original.min(), X_original.max(), 200).reshape(-1, 1)
    y_curve = np.array([forward_pass(network, x)[0] for x in X_curve])
    
    ax.plot(X_curve, y_curve, label='Curva Aprendida pela Rede', color='red', linewidth=2)

    ax.set_title('Resultado do Curve Fitting', fontsize=16)
    ax.set_xlabel('Entrada (X)')
    ax.set_ylabel('Saída (Y)')
    ax.legend()
    ax.grid(True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, "plot_5_curve_fit.png")
    plt.savefig(filename)
    print(f"Gráfico de Curve Fitting salvo como '{filename}'")
    plt.close(fig)


def plot_embeddings(words, embeddings):
    """Plota os embeddings das palavras em 2D e salva a imagem em arquivo."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embeddings[:, 0], embeddings[:, 1], color="orange", edgecolor="black", s=100)

    for i, word in enumerate(words):
        ax.text(embeddings[i, 0], embeddings[i, 1], word, fontsize=12, ha='center', va='bottom')

    ax.set_title("Word Embeddings (2D)", fontsize=14)
    ax.set_xlabel("Dimensão 1")
    ax.set_ylabel("Dimensão 2")
    ax.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, "plot_6_word_embeddings.png")
    plt.savefig(filename)
    print(f"Gráfico de embeddings salvo como '{filename}'")
    plt.close(fig)


def plot_embeddings_3d(words, embeddings):
    """Plota os embeddings das palavras em 3D de vários ângulos e salva em arquivos."""
    if embeddings.shape[1] < 3:
        print("Os embeddings precisam ter pelo menos 3 dimensões para visualização 3D.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    angles = [-60, 0, 60, 120]
    for i, angle in enumerate(angles):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2],
            color='orange',
            edgecolor='black',
            s=120,
            depthshade=True
        )

        # Adiciona o nome de cada palavra no ponto
        for j, word in enumerate(words):
            ax.text(
                embeddings[j, 0],
                embeddings[j, 1],
                embeddings[j, 2],
                word,
                fontsize=12,
                ha='center',
                va='bottom'
            )

        ax.set_title("Word Embeddings (3D)", fontsize=14)
        ax.set_xlabel("Dimensão 1")
        ax.set_ylabel("Dimensão 2")
        ax.set_zlabel("Dimensão 3")

        ax.view_init(elev=30, azim=angle)

        filename = os.path.join(OUTPUT_DIR, f"plot_7_word_embeddings_3d_angle_{i+1}.png")
        plt.savefig(filename)
        print(f"Gráfico de embeddings 3D (ângulo {angle}°) salvo como '{filename}'")
        plt.close(fig)
