import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import random

BG_COLOR = "#454545"
TITLE_FONT = ("Arial", 32, "bold")
IMAGE_SIZE = 900


def screen_size(root):
    """
    Gets the width and height of the user's screen.

    :param root: the main Tkinter window
    :return: the screen width and screen height
    """

    # get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # return the screen dimensions so other functions can use them
    return screen_width, screen_height


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


def return_label(root, image_num, screen_height):
    """
    Creates and places the label showing the correct digit.

    :param root: the main Tkinter window
    :param image_num: the correct digit for the current MNIST image
    :param screen_height: the height of the user's screen
    :return: the digit label so it can be updated later
    """

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

    # return the label so update_image() can change its text later
    return digit_label


def return_image(root, pil_image, screen_height):
    """
    Creates and places the image label that displays the MNIST image.

    :param root: the main Tkinter window
    :param pil_image: the prepared Pillow image
    :param screen_height: the height of the user's screen
    :return: the image label so it can be updated later
    """

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

    # return the image label so update_image() can change the image later
    return image_label


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


def update_image(training_data, image_label, digit_label):
    """
    Gets a new random MNIST image and updates the existing image and digit label.

    :param training_data: the MNIST training data
    :param image_label: the label currently displaying the MNIST image
    :param digit_label: the label currently displaying the correct digit
    :return: nothing
    """

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


def next_image_button(root, training_data, image_label, digit_label, screen_height):
    """
    Creates and places the button that shows a new random MNIST image.

    :param root: the main Tkinter window
    :param training_data: the MNIST training data
    :param image_label: the label currently displaying the MNIST image
    :param digit_label: the label currently displaying the correct digit
    :param screen_height: the height of the user's screen
    :return: nothing
    """

    # create a button that calls update_image() when clicked
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


def show_ui(training_data):
    """
    Creates the full MNIST viewer window and starts the Tkinter event loop.

    :param training_data: the MNIST training data
    :return: nothing
    """

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

    # get the first random image and correct number
    pil_image, image_num = prepare_random_mnist_image(training_data)

    # create the image label
    image_label = return_image(root, pil_image, screen_height)

    # create the correct-number label
    digit_label = return_label(root, image_num, screen_height)

    # create button that gets another random image
    next_image_button(root, training_data, image_label, digit_label, screen_height)

    # run the window
    root.mainloop()