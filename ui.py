import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import random

image_size = 900

def show_ui(training_data):
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.title("Mnist Training window")
    root.configure(
        bg="#454545"
    )

    screen_height = root.winfo_screenheight()

    random_index = random.randint(0, len(training_data) - 1)

    image_num = np.argmax(training_data[random_index][1])

    image = training_data[random_index][0]

    image = image.reshape(28, 28)

    image = (image * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    pil_image = pil_image.resize((image_size, image_size), Image.NEAREST)

    tk_image = ImageTk.PhotoImage(pil_image)

    image_label = tk.Label(
        root,
        image=tk_image
    )

    image_label.image = tk_image

    image_label.place(
        x=100,
        y=(screen_height - image_size) / 2
    )

    lbl = tk.Label(
        root,
        text="Mnist Training",
        font=("Arial", 32, "bold"),
        bg="#454545"
    )
    lbl.pack()

    lbl = tk.Label(
        root,
        text=image_num,
        font=("Arial", 32, "bold"),
        bg="#454545"
    )
    lbl.place(
        x=1700,
        y=(screen_height - image_size) / 2
    )

    exit_button = tk.Button(
        root,
        text="Exit",
        command=root.destroy
    )
    exit_button.place(
        x=2192
    )

    root.mainloop()