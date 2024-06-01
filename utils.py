import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import pickle
from PIL import Image, ImageFilter
from ipywidgets import HBox, VBox, Button, Label, Output
from ipycanvas import Canvas, hold_canvas

def load_model(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)

def save_model(s, fn):
    with open(fn, "wb") as f:
        pickle.dump(s, f)

def plot_mnist(X, y, n_rows=4, n_cols=10):
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X[index].reshape(28,28), cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(y[index], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()


def create_mnist_canvas(proc_image):
    width = 280
    height = 280

    output = Output()
    canvas = Canvas(width=width, height=height, sync_image_data=True)
    canvas_thumb = Canvas(width=28, height=28, layout={"width": "28px", "max-width": "28px"})

    drawing = False
    position = None
    shape = []


    def on_mouse_down(x, y):
        nonlocal drawing
        nonlocal position
        nonlocal shape

        drawing = True
        position = (x, y)
        shape = [position]


    def on_mouse_move(x, y):
        nonlocal drawing
        nonlocal position
        nonlocal shape

        if not drawing:
            return

        with hold_canvas():
            canvas.stroke_line(position[0], position[1], x, y)
            position = (x, y)

        shape.append(position)


    def on_mouse_up(x, y):
        nonlocal drawing
        nonlocal position
        nonlocal shape

        drawing = False

        with hold_canvas():
            canvas.stroke_line(position[0], position[1], x, y)

        shape = []


    canvas.on_mouse_down(on_mouse_down)
    canvas.on_mouse_move(on_mouse_move)
    canvas.on_mouse_up(on_mouse_up)

    canvas.stroke_style = "#000000"
    canvas.line_width = 15
    canvas.line_join = "round"
    canvas.line_cap = "round"
    #canvas.filter = "blur(1px)"
    canvas.miter_limit = 10

    clear_button = Button(description="clear")
    clear_button.on_click(lambda _: canvas.clear())

    result_text = Label()

    def on_process(_):
        try:
            canvas.flush()

            img = Image.open(BytesIO(canvas.image_data)).filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.GaussianBlur(4))
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img = np.asarray(img, dtype=np.float32)

            canvas_thumb.clear()
            canvas_thumb.put_image_data(img)

            img = img.max(axis=-1)

            result_text.value = str(proc_image(img))
            canvas.clear()
        except Exception as e:
            with output:
                print(e)

    process_button = Button(description="process")
    process_button.on_click(on_process)

    ctrls_box = VBox((clear_button, process_button, result_text))
    return VBox((HBox((canvas, ctrls_box)), canvas_thumb, output))


