import mnist_loader
import ui as ui
from network import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

INPUT_NEURON = 784
OUTPUT_NEURON = 10

current_network = None
current_training_id = 0

def train_model(log_callback, hidden_neuron, epoch, mini_batch_size, eta, lmbda, image_shift, cost_function, output_activation, weight_init, save_model, model_name):
    global current_network, current_training_id

    current_training_id += 1
    my_training_id = current_training_id

    current_network = network(
        [INPUT_NEURON, hidden_neuron,
        OUTPUT_NEURON],
        weight_initialization=weight_init,
        output_activation=output_activation
    )

    training_finished = current_network.SGD(
        training_data=training_data,
        epochs=epoch,
        mini_batch_size=mini_batch_size,
        eta=eta,
        lmbda=lmbda,
        cost_function=cost_function,
        image_shift=image_shift,
        test_data=test_data,
        training_id=my_training_id,
        should_stop=lambda training_id: training_id != current_training_id,
        log_callback=log_callback
    )

    if save_model and training_finished:
        filename = current_network.save(model_name)

        if log_callback:
            log_callback(f"Model saved as: {filename}")


# enable to show ui
ui.show_ui(training_data, train_model)