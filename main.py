import mnist_loader
import ui as ui
from network import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

INPUT_NEURON = 784
HIDDEN_NEURON = 30
OUTPUT_NEURON = 10
# options: random, improved
WEIGHT_INITIALIZATION = "improved"

net = network([INPUT_NEURON, HIDDEN_NEURON, OUTPUT_NEURON], weight_initialization=WEIGHT_INITIALIZATION)

# enable to show ui
#ui.show_ui(training_data)