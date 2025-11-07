import numpy as np
import neuralnet as neuralnet
import serialization as serialization
import show_graphic as show_graphic
import activation_functions as activation_functions

def main():
    # --- 1. Dataset ---
    # Vocabulary
    words = ["cat", "dog", "car"]

    # One-hot encoding
    X = np.eye(len(words))  # [[1,0,0], [0,1,0], [0,0,1]]

    # Fake semantic relationships (for example):
    # cats and dogs are similar, car is far from both
    y = np.array([
        [0.9, 0.1],  # cat
        [0.8, 0.2],  # dog
        [0.1, 0.9],  # car
    ])

    # --- 2. Build the network ---
    # Input: 3 (one-hot)
    # Hidden: 4 neurons (optional)
    # Output: 2 (embedding)
    network = neuralnet.build_network([3, 4, 2])

    # Output layer activation: linear (since embeddings are just numbers)
    output_layer = network[-1]
    for neuron in output_layer:
        neuron["activation_func"] = activation_functions.linear
        neuron["activation_deriv"] = activation_functions.linear_derivative

    # --- 3. Train ---
    print("Training small embedding network...")
    trained_net, history = neuralnet.train_network(
        network=network,
        input_value=X,
        output_value=y,
        epochs=2000,
        learning_rate=0.05
    )

    print("Training done.\n")

    # --- 4. Save the model ---
    serialization.save_network_state(trained_net, filename="word_embedding_model")

    # --- 5. Visualize embeddings ---
    print("Learned embeddings:")
    for i, word in enumerate(words):
        embedding = neuralnet.forward_pass(trained_net, X[i])
        print(f"{word}: {embedding}")

    # Optional: plot the embeddings in 2D
    embeddings = np.array([neuralnet.forward_pass(trained_net, X[i]) for i in range(len(words))])
    show_graphic.plot_embeddings(words, embeddings)

    # --- 7. Plot embeddings 3D ---
    print("Generating 3D embedding plots (rotated views)...")
    # Expand to 3D (add a dummy z-axis = 0)
    embeddings_3d = np.hstack([embeddings, np.zeros((embeddings.shape[0], 1))])
    show_graphic.plot_embeddings_3d(words, embeddings_3d)

if __name__ == "__main__":
    main()
