import matplotlib.pyplot as plt
import numpy as np

def view_img_classify(img, ps, version="MNIST"):
    """A function for viewing an image classification very visually

    :img: A tensor representing a 28x28 image
    :ps: the probabilities for each class
    :version: what type of image it is, one of "MNIST",
    :returns: None

    """
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,9))
    ax1.imshow(img.view(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))

    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))

    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


