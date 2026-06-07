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
IMAGE_SHIFT = 1

# options: random, improved
WEIGHT_INITIALIZATION = "improved"

# options: quadratic, cross_entropy
COST_FUNCTION = "cross_entropy"

net = network([INPUT_NEURON, HIDDEN_NEURON, OUTPUT_NEURON], weight_initialization=WEIGHT_INITIALIZATION)

net.SGD(
    training_data=training_data,
    epochs=EPOCHS,
    mini_batch_size=MINI_BATCH_SIZE,
    eta=ETA,
    cost_function=COST_FUNCTION,
    image_shift=IMAGE_SHIFT,
    test_data=test_data,
)

# enable to show ui
#ui.show_ui(training_data)