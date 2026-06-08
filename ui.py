import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import random
import threading

BG_COLOR = "#454545"
TITLE_FONT = ("Arial", 32, "bold")
IMAGE_SIZE = 900
menu_buttons = []


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

    return console


def write_to_console(console, message):
    console.insert(tk.END, message + "\n")
    console.see(tk.END)


def clear_menu(menu_buttons):
    for button in menu_buttons:
        button.destroy()


def screen_size(root):
    # get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # return the screen dimensions so other functions can use them
    return screen_width, screen_height


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


def create_title(root):
    """
    Creates and places the title text at the top of the window.

    :param root: the main Tkinter window
    :return: nothing
    """

    # create the title label
    title = tk.Label(
        root,
        text="Mnist Training",
        font=TITLE_FONT,
        bg=BG_COLOR
    )

    # place the title in the window
    title.pack()


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

    return image_label, digit_label


def next_image_button(root, training_data, image_label, digit_label, screen_height):# create a button that calls update_image() when clicked
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


def show_image_window(root, training_data, screen_height):
    image_label, digit_label = display_first_image(root, training_data, screen_height)

    next_image_button(root, training_data,image_label, digit_label, screen_height)


def image_view_button(root, training_data, screen_height, menu_buttons):
    image_view_button = tk.Button(
        root,
        text="Show Images",
        width=20,
        height=3,
        font=TITLE_FONT,
        command=lambda: (clear_menu(menu_buttons), show_image_window(root, training_data, screen_height))
    )

    menu_buttons.append(image_view_button)

    # place the exit button in the window
    image_view_button.place(
        x=300,
        y=400
    )


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


def open_training_page(root, menu_buttons, train_model):
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

    clear_menu(menu_buttons)

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


def training_page_button(root, screen_width, menu_buttons, train_model):
    training_page_button = tk.Button(
        root,
        text="Train Model",
        width=20,
        height=3,
        font=TITLE_FONT,
        command=lambda: open_training_page(
            root,
            menu_buttons,
            train_model
        )
    )

    menu_buttons.append(training_page_button)

    # place the exit button in the window
    training_page_button.place(
        relx=1.0,
        x=-300,
        y=400,
        anchor="ne"
    )


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
    create_title(root)

    # create image view option button
    image_view_button(root, training_data, screen_height, menu_buttons)

    # create train model button
    training_page_button(root, screen_width, menu_buttons, train_model)

    # run the window
    root.mainloop()