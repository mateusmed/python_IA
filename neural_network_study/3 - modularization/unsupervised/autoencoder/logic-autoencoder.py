import numpy as np
import neuralnet as neuralnet
import serialization as serialization
import show_graphic as show_graphic
import activation_functions as activation_functions

def main():
    # --- 1. Frases de exemplo ---
    sentences = [
        "gato dorme",
        "gato mia",
        "carro anda",
        "carro buzina",
    ]

    # --- 2. Vetorização simples (Bag of Words) ---
    # Cria um vocabulário com todas as palavras únicas
    vocab = sorted(set(" ".join(sentences).split()))
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Converte frases em vetores binários
    def vectorize(sentence):
        vec = np.zeros(len(vocab))
        for word in sentence.split():
            vec[vocab_index[word]] = 1
        return vec

    X = np.array([vectorize(s) for s in sentences])
    print("Vocabulário:", vocab)
    print("Entradas (vetores):\n", X)


    # --- 3. Cria o autoencoder ---
    # Exemplo: entrada N palavras → camada oculta 2 → saída N palavras
    input_size = len(vocab)
    network = neuralnet.build_network([input_size, 2, input_size])

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
        epochs=5000,
        learning_rate=0.05
    )

    # salva e plota erros
    show_graphic.plot_errors_per_epoch(history)

    print("Training done.\n")


    # --- 6. Extrai os embeddings (camada escondida) ---
    print("\nEmbeddings aprendidos:")
    embeddings = []
    for i, x in enumerate(X):
        outputs = neuralnet.forward_pass(trained_net, x, return_all_layers=True)
        # todo melhorar (camada intermediaria)
        hidden = outputs[1]  # camada intermediária
        embeddings.append(hidden)
        print(f"{sentences[i]:<15} -> {hidden}")

    embeddings = np.array(embeddings)

    # todo depois tentar fazer correlação não das sentensas e sim das palavras soltas
    # --- 7. Mostra os embeddings em gráfico 2D ---
    show_graphic.plot_embeddings(sentences, embeddings)


    # --- 7. Plot embeddings 3D ---
    print("Generating 3D embedding plots (rotated views)...")
    # Expand to 3D (add a dummy z-axis = 0)
    embeddings_3d = np.hstack([embeddings, np.zeros((embeddings.shape[0], 1))])
    show_graphic.plot_embeddings_3d(sentences, embeddings_3d)

    # --- 8. Salva o modelo (opcional) ---
    serialization.save_network_state(trained_net, filename="text_autoencoder_model")


if __name__ == "__main__":
    main()
