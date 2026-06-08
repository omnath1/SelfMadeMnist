import mnist_loader
import ui as ui
from network import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

INPUT_NEURON = 784
HIDDEN_NEURON = 128
OUTPUT_NEURON = 10
EPOCHS = 30
MINI_BATCH_SIZE = 10
ETA = 0.5
LMBDA = 5.0 # 0.0 == L2 off
IMAGE_SHIFT = 0

# options: random, improved
WEIGHT_INITIALIZATION = "improved"

# options: quadratic, cross_entropy
COST_FUNCTION = "cross_entropy"

# options: sigmoid, softmax
OUTPUT_ACTIVATION = "softmax"

current_network = None
current_training_id = 0

def train_model():
    global current_network, current_training_id

    current_training_id += 1
    my_training_id = current_training_id

    current_network = network(
        [INPUT_NEURON, HIDDEN_NEURON,
        OUTPUT_NEURON],
        weight_initialization=WEIGHT_INITIALIZATION,
        output_activation=OUTPUT_ACTIVATION
    )

    current_network.SGD(
        training_data=training_data,
        epochs=EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        eta=ETA,
        lmbda=LMBDA,
        cost_function=COST_FUNCTION,
        image_shift=IMAGE_SHIFT,
        test_data=test_data,
        training_id=my_training_id,
        should_stop=lambda training_id: training_id != current_training_id
    )


# enable to show ui
ui.show_ui(training_data, train_model)

# TODO: add a button to the main menu that
#  starts the training and shows epoch progress


