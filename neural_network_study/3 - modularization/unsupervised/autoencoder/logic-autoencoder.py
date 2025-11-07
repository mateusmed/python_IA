import numpy as np
import neuralnet as neuralnet
import serialization as serialization
import show_graphic as show_graphic
import activation_functions as activation_functions

def main():
    # dataset simples de exemplo
    X = np.eye(3)  # [ [1,0,0], [0,1,0], [0,0,1] ]

    # arquitetura do autoencoder
    # entrada 3 → escondida 2 → saída 3 (reconstrução)
    network = neuralnet.build_network([3, 2, 3])

    # camada de saída linear (sem restrição de faixa)
    output_layer = network[-1]
    for neuron in output_layer:
        neuron["activation_func"] = activation_functions.linear
        neuron["activation_deriv"] = activation_functions.linear_derivative

    # treino auto-supervisionado (entrada = saída)
    trained_net, history = neuralnet.train_network(
        network=network,
        input_value=X,
        output_value=X,
        epochs=3000,
        learning_rate=0.05
    )

    # salva e plota erros
    show_graphic.plot_errors_per_epoch(history)

    print("Training done.\n")

    # --- 4. Save the model ---
    serialization.save_network_state(trained_net, filename="word_embedding_model")

    # obtém o embedding (saída da camada intermediária)
    print("\nEmbeddings aprendidos:")
    for i, x in enumerate(X):
        outputs = neuralnet.forward_pass(trained_net, x, return_all=True)
        hidden = outputs[1]  # camada do meio (embedding)
        print(f"Entrada {i}: {hidden}")

    # plota embeddings em 2D
    words = ["cat", "dog", "car"]
    embeddings = np.array([neuralnet.forward_pass(trained_net, x, return_all=True)[1] for x in X])
    show_graphic.plot_embeddings(words, embeddings)

    # --- 7. Plot embeddings 3D ---
    print("Generating 3D embedding plots (rotated views)...")
    # Expand to 3D (add a dummy z-axis = 0)
    embeddings_3d = np.hstack([embeddings, np.zeros((embeddings.shape[0], 1))])
    show_graphic.plot_embeddings_3d(words, embeddings_3d)


if __name__ == "__main__":
    main()
