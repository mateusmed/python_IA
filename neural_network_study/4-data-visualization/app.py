from flask import Flask, jsonify, render_template
from flask_cors import CORS
import numpy as np
import neuralnet
import activation_functions
import os

app = Flask(__name__)
CORS(app)

# Initialize a standard XOR network
# Structure: 2 inputs -> 4 hidden -> 1 output
LAYER_STRUCTURE = [2, 4, 1]
network = neuralnet.build_network(LAYER_STRUCTURE)

# Sample training data for XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train for a few epochs just to have some non-random weights
neuralnet.train_network(network, X, y, epochs=100, learning_rate=0.1)

def serialize_network(net):
    serializable_net = []
    for layer in net:
        serializable_layer = []
        for neuron in layer:
            n = neuron.copy()
            # Convert numpy arrays to lists
            n["weights"] = n["weights"].tolist()
            # Convert functions to names for JSON serialization
            n["activation_func"] = "sigmoid" # simplifying for the demo
            n["activation_deriv"] = "sigmoid_derivative"
            
            # Convert log items (numpy arrays) to lists
            if n["log"]:
                serializable_log = []
                for entry in n["log"][-10:]: # limit to last 10 entries
                    entry_copy = entry.copy()
                    for key, val in entry_copy.items():
                        if isinstance(val, np.ndarray):
                            entry_copy[key] = val.tolist()
                    serializable_log.append(entry_copy)
                n["log"] = serializable_log
            else:
                n["log"] = []
                
            serializable_layer.append(n)
        serializable_net.append(serializable_layer)
    return serializable_net

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/network")
def get_network():
    return jsonify({
        "structure": LAYER_STRUCTURE,
        "layers": serialize_network(network)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
