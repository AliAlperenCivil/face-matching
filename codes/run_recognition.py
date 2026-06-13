# Train on for_training and recognise one image from for_testing

import os
import numpy as np
import matplotlib.pyplot as plt

from eigenfaces import train_eigenfaces, test_eigenfaces

# folders (edit these two lines if your paths are different)
HERE = os.path.dirname(os.path.abspath(__file__))
# data folders may sit next to this script or one level up (e.g. in a codes/ folder)
ROOT = HERE if os.path.isdir(os.path.join(HERE, "for_training")) else os.path.dirname(HERE)
TRAINING_FOLDER = os.path.join(ROOT, "for_training")
TEST_IMAGE = os.path.join(ROOT, "for_testing", "whitewoman_t.pgm")


def main():
    # Step 1 - training
    model = train_eigenfaces(TRAINING_FOLDER)
    print("Trained on %d faces, size %dx%d" % (model["M"], model["H"], model["W"]))
    print("Training order:", model["names"])

    # Step 2 - testing
    index, name, distances = test_eigenfaces(TEST_IMAGE, model)

    # print the distances scaled by 1e9, same style as MATLAB
    print("\ndist (x 1e9):")
    for i, d in enumerate(distances):
        marker = "  <-- minimum" if i == index else ""
        print("  %-18s %.4f%s" % (model["names"][i], d / 1e9, marker))
    print("\nRecognised as: %s (index %d)" % (name, index))

    # show the test face next to the recognised training face
    test_face = np.asarray(plt.imread(TEST_IMAGE))
    recognised_face = model["images"][index]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(test_face, cmap="gray")
    axes[0].set_title("Test face")
    axes[0].axis("off")
    axes[1].imshow(recognised_face, cmap="gray")
    axes[1].set_title("Recognized face")
    axes[1].axis("off")
    fig.suptitle("%s  ->  %s" % (os.path.basename(TEST_IMAGE), name))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()