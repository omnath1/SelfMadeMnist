import mnist_loader
import ui as ui
from network import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

INPUT_NEURON = 784
OUTPUT_NEURON = 10

# options: random, improved
WEIGHT_INITIALIZATION = "improved"

# options: quadratic, cross_entropy
COST_FUNCTION = "cross_entropy"

# options: sigmoid, softmax
OUTPUT_ACTIVATION = "softmax"

current_network = None
current_training_id = 0

def train_model(log_callback, hidden_neuron, epoch, mini_batch_size, eta, lmbda, image_shift):
    global current_network, current_training_id

    current_training_id += 1
    my_training_id = current_training_id

    current_network = network(
        [INPUT_NEURON, hidden_neuron,
        OUTPUT_NEURON],
        weight_initialization=WEIGHT_INITIALIZATION,
        output_activation=OUTPUT_ACTIVATION
    )

    current_network.SGD(
        training_data=training_data,
        epochs=epoch,
        mini_batch_size=mini_batch_size,
        eta=eta,
        lmbda=lmbda,
        cost_function=COST_FUNCTION,
        image_shift=image_shift,
        test_data=test_data,
        training_id=my_training_id,
        should_stop=lambda training_id: training_id != current_training_id,
        log_callback=log_callback
    )


# enable to show ui
ui.show_ui(training_data, train_model)