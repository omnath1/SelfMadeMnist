import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import random

BG_COLOR = "#454545"
TITLE_FONT = ("Arial", 32, "bold")
IMAGE_SIZE = 900

def screen_size(root):

    # get screen dimentions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    return screen_width, screen_height

def prepare_random_mnist_image(training_data):

    # get a random number
    random_index = random.randint(0, len(training_data) - 1)

    # get image and label at position of random num
    image, label_vector = training_data[random_index]
    image_num = np.argmax(label_vector)

    # reshape the image back into a 28 * 28 image
    image = image.reshape(28, 28)

    # multiplies all values from 0.0-1.0 to 0-255
    image = (image * 255).astype(np.uint8)

    # turn image into a pillow image
    pil_image = Image.fromarray(image)

    # reshape the image to any size while keeping the 28*28 aesthetic
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    return pil_image, image_num

def create_title(root):
    # place the title in the window
    title = tk.Label(
        root,
        text="Mnist Training",
        font=TITLE_FONT,
        bg=BG_COLOR
    )
    title.pack()

def create_label(root, image_num, screen_height):
    # place label in the window
    digit_label = tk.Label(
        root,
        text=image_num,
        font=TITLE_FONT,
        bg=BG_COLOR
    )
    digit_label.place(
        x=1700,
        y=(screen_height - IMAGE_SIZE) / 2
    )

def create_image(root, pil_image, screen_height):
    # convert Pillow image to a Tkinter-compatible image
    tk_image = ImageTk.PhotoImage(pil_image)

    # place image in the window
    image_label = tk.Label(
        root,
        image=tk_image
    )

    image_label.image = tk_image

    image_label.place(
        x=100,
        y=(screen_height - IMAGE_SIZE) / 2
    )

def create_exit_button(root):
    # exit button
    exit_button = tk.Button(
        root,
        text="Exit",
        command=root.destroy
    )
    exit_button.place(
        x=2192
    )

def show_ui(training_data):
    # create a window
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.title("Mnist Training window")
    root.configure(
        bg=BG_COLOR
    )

    # get screen dimentions
    screen_width, screen_height = screen_size(root)

    # create exit button
    create_exit_button(root)

    # show the title
    create_title(root)

    # call image function
    pil_image, image_num = prepare_random_mnist_image(training_data)

    # show the image
    create_image(root, pil_image, screen_height)

    # create label
    create_label(root, image_num, screen_height)

    # run the window
    root.mainloop()