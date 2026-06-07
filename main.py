import mnist_loader
import ui as ui

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

ui.show_ui(training_data)