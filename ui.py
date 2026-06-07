import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import random

def show_ui(training_data):
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.title("Mnist Training window")
    root.configure(
        bg="#454545"
    )

    # random_index = random.randint(0, len(training_data) - 1)

    image = training_data[0][0]

    image = image.reshape(28, 28)

    image = (image * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    pil_image = pil_image.resize((280, 280))

    tk_image = ImageTk.PhotoImage(pil_image)

    image_label = tk.Label(
        root,
        image=tk_image
    )

    image_label.image = tk_image

    image_label.place(
        x=0,
        y=0
    )

    lbl = tk.Label(
        root,
        text="Mnist Training",
        font=("Arial", 32, "bold"),
        bg="#454545"
    )
    lbl.pack()

    exit_button = tk.Button(
        root,
        text="Exit",
        command=root.destroy
    )
    exit_button.place(
        x=2192
    )

    root.mainloop()