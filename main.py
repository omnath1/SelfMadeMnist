import mnist_loader
import ui as ui
from network import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

INPUT_NEURON = 784
HIDDEN_NEURON = 30
OUTPUT_NEURON = 10
EPOCHS = 30
MINI_BATCH_SIZE = 10
ETA = 0.5

# options: random, improved
WEIGHT_INITIALIZATION = "random"

net = network([INPUT_NEURON, HIDDEN_NEURON, OUTPUT_NEURON], weight_initialization=WEIGHT_INITIALIZATION)

net.SGD(
    training_data=training_data,
    epochs=EPOCHS,
    mini_batch_size=MINI_BATCH_SIZE,
    eta=ETA,
    test_data=test_data
)

# enable to show ui
#ui.show_ui(training_data)