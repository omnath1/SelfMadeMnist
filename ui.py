# imports
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import random
import threading
import os
from network import network

# Constants / global variables
BG_COLOR = "#454545"
TITLE_FONT = ("Arial", 32, "bold")
current_page_widgets = []
GRID_SIZE = 28
PIXEL_SIZE = 25
IMAGE_SIZE = GRID_SIZE * PIXEL_SIZE
current_loaded_model = None
prediction_labels = []
guess_label = None
loaded_model_label = None

# General helper functions
def clear_current_page():
    for widget in current_page_widgets:
        widget.destroy()

    current_page_widgets.clear()


def screen_size(root):
    # get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # return the screen dimensions so other functions can use them
    return screen_width, screen_height


def write_to_console(console, message):
    console.insert(tk.END, message + "\n")
    console.see(tk.END)


# General reusable UI functions
def create_exit_button(root):
    """
    Creates and places the exit button.

    :param root: the main Tkinter window
    :return: nothing
    """

    # create the exit button
    exit_button = tk.Button(
        root,
        text="Exit",
        command=root.destroy
    )

    # place the exit button in the window
    exit_button.place(
        x=2192
    )


def create_back_button(root, training_data, train_model):
    back_button = tk.Button(
        root,
        text="Back",
        width=12,
        height=2,
        command=lambda: show_main_menu(root, training_data, train_model)
    )

    back_button.place(
        x=0,
        y=0
    )

    current_page_widgets.append(back_button)


# Title functions
def create_menu_title(root):# create the title label
    title = tk.Label(
        root,
        text="Mnist Training Manager",
        font=TITLE_FONT,
        bg=BG_COLOR
    )

    # place the title in the window
    title.pack()

    current_page_widgets.append(title)


def create_image_title(root):# create the title label
    title = tk.Label(
        root,
        text="Mnist Image Viewer",
        font=TITLE_FONT,
        bg=BG_COLOR
    )

    # place the title in the window
    title.pack()

    current_page_widgets.append(title)


def create_train_title(root):# create the title label
    title = tk.Label(
        root,
        text="Mnist Model Training",
        font=TITLE_FONT,
        bg=BG_COLOR
    )

    # place the title in the window
    title.pack()

    current_page_widgets.append(title)


def create_test_title(root):# create the title label
    title = tk.Label(
        root,
        text="Mnist Model Tester",
        font=TITLE_FONT,
        bg=BG_COLOR
    )

    # place the title in the window
    title.pack()

    current_page_widgets.append(title)


# Main menu functions
def image_view_button(root, training_data, screen_height, current_page_widgets, train_model):
    image_view_button = tk.Button(
        root,
        text="Show Images",
        width=20,
        height=3,
        font=TITLE_FONT,
        command=lambda: (
            clear_current_page(),
            show_image_window(root, training_data, screen_height, current_page_widgets, train_model)
        )
    )

    current_page_widgets.append(image_view_button)

    # place the exit button in the window
    image_view_button.place(
        x=300,
        y=400
    )


def training_page_button(root, training_data, screen_width, current_page_widgets, train_model):
    training_page_button = tk.Button(
        root,
        text="Train Model",
        width=20,
        height=3,
        font=TITLE_FONT,
        command=lambda: open_training_page(
            root,
            training_data,
            current_page_widgets,
            train_model
        )
    )

    current_page_widgets.append(training_page_button)

    # place the exit button in the window
    training_page_button.place(
        relx=1.0,
        x=-300,
        y=400,
        anchor="ne"
    )


def test_page_button(root, training_data, train_model, screen_height):
    test_page_button = tk.Button(
        root,
        text="Test Model",
        width=20,
        height=3,
        font=TITLE_FONT,
        command=lambda: open_test_page(
            root,
            training_data,
            train_model,
            screen_height
        )
    )

    current_page_widgets.append(test_page_button)

    # place the exit button in the window
    test_page_button.place(
        x=300,
        y=800
    )


def show_main_menu(root, training_data, train_model):
    screen_width, screen_height = screen_size(root)

    clear_current_page()

    image_view_button(
        root,
        training_data,
        screen_height,
        current_page_widgets,
        train_model
    )

    training_page_button(
        root,
        training_data,
        screen_width,
        current_page_widgets,
        train_model
    )

    test_page_button(
        root,
        training_data,
        train_model,
        screen_height
    )

    create_menu_title(
        root
    )


# Image viewer page
def prepare_random_mnist_image(training_data):
    """
    Picks a random MNIST image from the training data and prepares it for Tkinter.

    :param training_data: the MNIST training data
    :return: a resized Pillow image and the correct digit label
    """

    # get a random number that can be used as a valid index in training_data
    random_index = random.randint(0, len(training_data) - 1)

    # get the image and label at the position of the random number
    # image is the 784 * 1 pixel vector
    # label_vector is the correct answer stored as a one-hot vector
    image, label_vector = training_data[random_index]

    # convert the one-hot vector back into the actual digit
    image_num = np.argmax(label_vector)

    # reshape the image back into a 28 * 28 image
    image = image.reshape(28, 28)

    # multiply all pixel values from 0.0-1.0 to 0-255
    # then convert them into normal image pixel values
    image = (image * 255).astype(np.uint8)

    # turn the NumPy array into a Pillow image
    pil_image = Image.fromarray(image)

    # resize the image while keeping the blocky 28 * 28 MNIST aesthetic
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    # return both the prepared image and the correct number
    return pil_image, image_num


def update_image(training_data, image_label, digit_label):
    # get a new random prepared MNIST image and its correct number
    pil_image, image_num = prepare_random_mnist_image(training_data)

    # convert the new Pillow image into a Tkinter-compatible image
    tk_image = ImageTk.PhotoImage(pil_image)

    # update the existing image label with the new image
    image_label.configure(image=tk_image)

    # keep a reference to the new image so Python does not delete it
    image_label.image = tk_image

    # update the existing digit label with the new correct number
    digit_label.configure(text=f"Correct number: {image_num}")


def display_first_image(root, training_data, screen_height):
    pil_image, image_num = prepare_random_mnist_image(training_data)

    # convert the Pillow image to a Tkinter-compatible image
    tk_image = ImageTk.PhotoImage(pil_image)

    # create the label that will display the image
    image_label = tk.Label(
        root,
        image=tk_image
    )

    # keep a reference to the image so Python does not delete it
    image_label.image = tk_image

    # place the image in the window
    image_label.place(
        x=100,
        y=(screen_height - IMAGE_SIZE) / 2
    )

    current_page_widgets.append(image_label)

    # create the label that shows the correct number
    digit_label = tk.Label(
        root,
        text=f"Correct number: {image_num}",
        font=TITLE_FONT,
        bg=BG_COLOR
    )

    # place the label below the image
    digit_label.place(
        x=300,
        y=((screen_height - IMAGE_SIZE) / 2) + IMAGE_SIZE + 3
    )

    current_page_widgets.append(digit_label)

    return image_label, digit_label


def next_image_button(root, training_data, image_label, digit_label, screen_height, current_page_widgets):# create a button that calls update_image() when clicked
    next_button = tk.Button(
        root,
        text="Next Image",
        width=12,
        height=2,
        command=lambda: update_image(training_data, image_label, digit_label)
    )

    # place the button under the image
    next_button.place(
        x=100,
        y=((screen_height - IMAGE_SIZE) / 2) + IMAGE_SIZE
    )

    current_page_widgets.append(next_button)


def show_image_window(root, training_data, screen_height, current_page_widgets, train_model):
    image_label, digit_label = display_first_image(root, training_data, screen_height)

    next_image_button(root, training_data,image_label, digit_label, screen_height, current_page_widgets)

    create_back_button(root, training_data, train_model)

    create_image_title(root)


# Training page helper widgets
def create_console(root):
    screen_width, screen_height = screen_size(root)

    console = tk.Text(
        root,
        font=("Arial", 18)
    )

    console.place(
        x=screen_width/2,
        y=200,
        width=(screen_width/2) - screen_width/10,
        height=screen_height *0.77
    )

    current_page_widgets.append(console)

    return console


def create_hidden_neuron_entry(root):
    hidden_neuron_label = tk.Label(
        root,
        text="Hidden neurons:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    hidden_neuron_label.place(
        relx=0.1,
        y=200
    )

    hidden_neuron_entry = tk.Entry(
        root,
        font=("Arial", 18),
        width=10
    )

    hidden_neuron_entry.insert(0, "128")

    hidden_neuron_entry.place(
        relx=0.1,
        y=250
    )

    current_page_widgets.append(hidden_neuron_entry)
    current_page_widgets.append(hidden_neuron_label)

    return hidden_neuron_entry


def create_epoch_entry(root):
    epoch_label = tk.Label(
        root,
        text="Epochs:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    epoch_label.place(
        relx=0.3,
        y=200
    )

    epoch_entry = tk.Entry(
        root,
        font=("Arial", 18),
        width=10
    )

    epoch_entry.insert(0, "30")

    epoch_entry.place(
        relx=0.3,
        y=250
    )

    current_page_widgets.append(epoch_entry)
    current_page_widgets.append(epoch_label)

    return epoch_entry


def create_mini_batch_size_entry(root):
    mini_batch_size_label = tk.Label(
        root,
        text="Mini batch size:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    mini_batch_size_label.place(
        relx=0.3,
        y=350
    )

    mini_batch_size_entry = tk.Entry(
        root,
        font=("Arial", 18),
        width=10
    )

    mini_batch_size_entry.insert(0, "10")

    mini_batch_size_entry.place(
        relx=0.3,
        y=400
    )

    current_page_widgets.append(mini_batch_size_entry)
    current_page_widgets.append(mini_batch_size_label)

    return mini_batch_size_entry


def create_eta_entry(root):
    eta_label = tk.Label(
        root,
        text="Eta:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    eta_label.place(
        relx=0.1,
        y=350
    )

    eta_entry = tk.Entry(
        root,
        font=("Arial", 18),
        width=10
    )

    eta_entry.insert(0, "0.5")

    eta_entry.place(
        relx=0.1,
        y=400
    )

    current_page_widgets.append(eta_label)
    current_page_widgets.append(eta_entry)

    return eta_entry


def create_lmbda_entry(root):
    lmbda_label = tk.Label(
        root,
        text="Lmbda:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    lmbda_label.place(
        relx=0.1,
        y=500
    )

    lmbda_entry = tk.Entry(
        root,
        font=("Arial", 18),
        width=10
    )

    lmbda_entry.insert(0, "5.0")

    lmbda_entry.place(
        relx=0.1,
        y=550
    )

    current_page_widgets.append(lmbda_label)
    current_page_widgets.append(lmbda_entry)

    return lmbda_entry


def create_image_shift_entry(root):
    image_shift_label = tk.Label(
        root,
        text="Image shift:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    image_shift_label.place(
        relx=0.3,
        y=500
    )

    image_shift_entry = tk.Entry(
        root,
        font=("Arial", 18),
        width=10
    )

    image_shift_entry.insert(0, "1")

    image_shift_entry.place(
        relx=0.3,
        y=550
    )

    current_page_widgets.append(image_shift_entry)
    current_page_widgets.append(image_shift_label)

    return image_shift_entry


def create_cost_function_dropdown(root):
    cost_function_label = tk.Label(
        root,
        text="Cost function:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    cost_function_label.place(
        relx=0.1,
        y=650
    )

    cost_function_var = tk.StringVar(root)
    cost_function_var.set("cross_entropy")

    cost_function_dropdown = tk.OptionMenu(
        root,
        cost_function_var,
        "quadratic",
        "cross_entropy"
    )

    cost_function_dropdown.place(
        relx=0.1,
        y=700
    )

    current_page_widgets.append(cost_function_dropdown)
    current_page_widgets.append(cost_function_label)

    return cost_function_var


def create_output_activation_dropdown(root):
    output_activation_label = tk.Label(
        root,
        text="Output activation:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    output_activation_label.place(
        relx=0.3,
        y=650
    )

    output_activation_var = tk.StringVar(root)
    output_activation_var.set("softmax")

    output_activation_dropdown = tk.OptionMenu(
        root,
        output_activation_var,
        "sigmoid",
        "softmax"
    )

    output_activation_dropdown.place(
        relx=0.3,
        y=700
    )

    current_page_widgets.append(output_activation_dropdown)
    current_page_widgets.append(output_activation_label)

    return output_activation_var


def create_weight_init_dropdown(root):
    weight_init_label = tk.Label(
        root,
        text="Weight Initialization:",
        font=("Arial", 18),
        bg=BG_COLOR
    )

    weight_init_label.place(
        relx=0.1,
        y=800
    )

    weight_init_var = tk.StringVar(root)
    weight_init_var.set("improved")

    output_activation_dropdown = tk.OptionMenu(
        root,
        weight_init_var,
        "improved",
        "random"
    )

    output_activation_dropdown.place(
        relx=0.1,
        y=850
    )

    current_page_widgets.append(output_activation_dropdown)
    current_page_widgets.append(weight_init_label)

    return weight_init_var


def create_save_model_toggle(root):
    save_model_var = tk.BooleanVar()

    save_model_toggle = tk.Checkbutton(
        root,
        text="Save model",
        variable=save_model_var,
        font=("Arial", 18),
        bg=BG_COLOR
    )

    save_model_toggle.place(
        relx=0.3,
        y=800
    )

    model_name_entry = tk.Entry(
        root,
        font=("Arial", 18),
        width=20
    )

    model_name_entry.insert(0, "model_name")

    model_name_entry.place(
        relx=0.3,
        y=850
    )

    current_page_widgets.append(model_name_entry)
    current_page_widgets.append(save_model_toggle)

    return save_model_var, model_name_entry


def train_model_button(root, train_model, console, hidden_neuron_entry, epoch_entry, mini_batch_size_entry, eta_entry, lmbda_entry, image_shift_entry, cost_function_entry, output_activation, weight_init, save_model_var, model_name_entry):
    train_model_button = tk.Button(
        root,
        text="Train Model",
        width=20,
        height=2,
        font=TITLE_FONT,
        command=lambda: threading.Thread(
            target=lambda: train_model(
                lambda message: write_to_console(console, message),
                int(hidden_neuron_entry.get()),
                int(epoch_entry.get()),
                int(mini_batch_size_entry.get()),
                float(eta_entry.get()),
                float(lmbda_entry.get()),
                int(image_shift_entry.get()),
                cost_function_entry.get(),
                output_activation.get(),
                weight_init.get(),
                save_model_var.get(),
                model_name_entry.get()
            ),
            daemon=True
        ).start()
    )

    # place the exit button in the window
    train_model_button.place(
        relx=0.1,
        y=1200
    )

    current_page_widgets.append(train_model_button)


# Training page opener
def open_training_page(root, training_data, current_page_widgets, train_model):
    clear_current_page()

    console = create_console(root)

    hidden_neuron_entry = create_hidden_neuron_entry(root)

    epoch_entry = create_epoch_entry(root)

    mini_batch_size_entry = create_mini_batch_size_entry(root)

    eta_entry = create_eta_entry(root)

    lmbda_entry = create_lmbda_entry(root)

    image_shift_entry = create_image_shift_entry(root)

    cost_function = create_cost_function_dropdown(root)

    output_activation = create_output_activation_dropdown(root)

    weight_init = create_weight_init_dropdown(root)

    save_model_var, model_name_entry = create_save_model_toggle(root)

    create_train_title(root)

    train_model_button(
        root, train_model,
        console,
        hidden_neuron_entry,
        epoch_entry,
        mini_batch_size_entry,
        eta_entry,
        lmbda_entry,
        image_shift_entry,
        cost_function,
        output_activation,
        weight_init,
        save_model_var,
        model_name_entry
    )

    create_back_button(root, training_data, train_model)


# Test / drawing page helper widgets
def create_drawing_canvas(root, screen_height):
    drawing_canvas = tk.Canvas(
        root,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        bg="black"
    )

    drawing_canvas.place(
        x=100,
        y=(screen_height - IMAGE_SIZE) / 2
    )

    current_page_widgets.append(drawing_canvas)

    return drawing_canvas


def setup_drawing(canvas):
    canvas.pixels = np.zeros((GRID_SIZE, GRID_SIZE))

    def draw_pixel(event):
        column = event.x // PIXEL_SIZE
        row = event.y // PIXEL_SIZE

        if 0 <= row < GRID_SIZE and 0 <= column < GRID_SIZE:
            canvas.pixels[row][column] = 1

            x1 = column * PIXEL_SIZE
            y1 = row * PIXEL_SIZE
            x2 = x1 + PIXEL_SIZE
            y2 = y1 + PIXEL_SIZE

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill="white",
                outline="white"
            )

            predict_canvas(canvas)

    canvas.bind("<Button-1>", draw_pixel)
    canvas.bind("<B1-Motion>", draw_pixel)

    return canvas.pixels


def create_clear_canvas_button(root, drawing_canvas, screen_height):
    clear_button = tk.Button(
        root,
        text="Clear",
        width=12,
        height=2,
        command=lambda: (
            drawing_canvas.delete("all"),
            drawing_canvas.pixels.fill(0),
            reset_prediction_table()
        )
    )

    clear_button.place(
        x=100,
        y=((screen_height - IMAGE_SIZE) / 2) + IMAGE_SIZE
    )

    current_page_widgets.append(clear_button)


def canvas_pixels_to_network_input(canvas):
    network_input = canvas.pixels.reshape(784, 1)

    return network_input


def predict_canvas(canvas):
    if current_loaded_model is None:
        print("No model loaded")
        return

    network_input = canvas_pixels_to_network_input(canvas)

    output = current_loaded_model.feedforward(network_input)

    update_prediction_table(output)


def get_saved_model_names():
    saved_model_folder = "saved_models"

    if not os.path.exists(saved_model_folder):
        return ["No saved models"]

    model_names = []

    for file_name in os.listdir(saved_model_folder):
        if file_name.endswith(".json"):
            model_names.append(file_name)

    if len(model_names) == 0:
        return ["No saved models"]

    return model_names


def create_saved_model_dropdown(root):
    model_names = get_saved_model_names()

    selected_model_var = tk.StringVar(root)
    selected_model_var.set(model_names[0])

    saved_model_dropdown = tk.OptionMenu(
        root,
        selected_model_var,
        *model_names
    )

    saved_model_dropdown.place(
        x=264,
        y=1115
    )

    current_page_widgets.append(saved_model_dropdown)

    return selected_model_var


def create_load_model_button(root, selected_model_var):
    load_model_button = tk.Button(
        root,
        text="Load Model",
        width=12,
        height=2,
        command=lambda: load_selected_model(selected_model_var)
    )

    load_model_button.place(
        x=640,
        y=1103
    )

    current_page_widgets.append(load_model_button)


def load_selected_model(selected_model_var):
    global current_loaded_model

    model_name = selected_model_var.get()

    if model_name == "No saved models":
        print("No model selected")
        return None

    model_path = os.path.join("saved_models", model_name)

    current_loaded_model = network.load(model_path)

    loaded_model_label.configure(
        text=f"Loaded model: {model_name}"
    )

    print("Loaded model:", model_name)

    return current_loaded_model


def create_prediction_table(root, screen_height):
    global prediction_labels, guess_label

    prediction_labels = []

    guess_label = tk.Label(
        root,
        text="Current guess: None",
        font=("Arial", 24, "bold"),
        bg=BG_COLOR
    )

    guess_label.place(
        x=1100,
        y=(screen_height - IMAGE_SIZE) / 2
    )

    current_page_widgets.append(guess_label)

    for digit in range(10):
        prediction_label = tk.Label(
            root,
            text=f"{digit}: 0%",
            font=("Arial", 18),
            bg=BG_COLOR
        )

        prediction_label.place(
            x=1100,
            y=((screen_height - IMAGE_SIZE) / 2) + 60 + digit * 35
        )

        prediction_labels.append(prediction_label)
        current_page_widgets.append(prediction_label)


def create_loaded_model_label(root, screen_height):
    global loaded_model_label

    loaded_model_label = tk.Label(
        root,
        text="Loaded model: None",
        font=("Arial", 18, "bold"),
        bg=BG_COLOR
    )

    loaded_model_label.place(
        x=100,
        y=((screen_height - IMAGE_SIZE) / 2) - 50
    )

    current_page_widgets.append(loaded_model_label)


def update_prediction_table(output):
    predicted_digit = np.argmax(output)

    guess_label.configure(
        text=f"Current guess: {predicted_digit}"
    )

    for digit in range(10):
        value = output[digit][0]

        prediction_labels[digit].configure(
            text=f"{digit}: {value*100:.2F}"
        )


def reset_prediction_table():
    guess_label.configure(
        text="Current guess: None"
    )

    for digit in range(10):
        prediction_labels[digit].configure(
            text=f"{digit}: 0%"
        )


# Test page opener
def open_test_page(root, training_data, train_model, screen_height):
    clear_current_page()

    create_test_title(root)

    create_loaded_model_label(root, screen_height)

    drawing_canvas = create_drawing_canvas(root, screen_height)

    setup_drawing(drawing_canvas)

    create_clear_canvas_button(root, drawing_canvas, screen_height)

    selected_model_var = create_saved_model_dropdown(root)

    create_load_model_button(
        root,
        selected_model_var
    )

    create_prediction_table(root, screen_height)

    create_back_button(root, training_data, train_model)


# 12. Main UI starter
def show_ui(training_data, train_model):
    # create the main window
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.title("Mnist Training window")
    root.configure(
        bg=BG_COLOR
    )

    # get screen dimensions
    screen_width, screen_height = screen_size(root)

    # create exit button
    create_exit_button(root)

    # show the title
    create_menu_title(root)

    # create image view option button
    image_view_button(
        root,
        training_data,
        screen_height,
        current_page_widgets,
        train_model
    )

    # create train model button
    training_page_button(
        root,
        training_data,
        screen_width,
        current_page_widgets,
        train_model
    )

    test_page_button(
        root,
        training_data,
        train_model,
        screen_height
    )

    # run the window
    root.mainloop()