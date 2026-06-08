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


def train_model_button(root, train_model, console):
    train_model_button = tk.Button(
        root,
        text="Train Model",
        width=20,
        height=2,
        font=TITLE_FONT,
        command=lambda: threading.Thread(
            target=lambda: train_model(lambda message: write_to_console(console, message)),
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

    clear_menu(menu_buttons)

    train_model_button(root, train_model, console)


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