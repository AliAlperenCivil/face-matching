# Train once and test several images, saving a result figure for each

import os
import numpy as np
import matplotlib.pyplot as plt

from eigenfaces import train_eigenfaces, test_eigenfaces

HERE = os.path.dirname(os.path.abspath(__file__))
# data folders may sit next to this script or one level up (e.g. in a codes/ folder)
ROOT = HERE if os.path.isdir(os.path.join(HERE, "for_training")) else os.path.dirname(HERE)
TRAINING_FOLDER = os.path.join(ROOT, "for_training")
TESTING_FOLDER = os.path.join(ROOT, "for_testing")
RESULTS_FOLDER = os.path.join(ROOT, "results")

# the test images we want to run (edit this list as you like)
TEST_IMAGES = [
    "girl_t.pgm",
    "spanishwoman_t.pgm",
    "brownwoman_t.pgm",
    "chinat_t.pgm",
    "bread_t.pgm",
]


# train once, then test and save a figure for every image in the list
def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    model = train_eigenfaces(TRAINING_FOLDER)
    print("Trained on %d faces, size %dx%d\n" % (model["M"], model["H"], model["W"]))

    for filename in TEST_IMAGES:
        test_path = os.path.join(TESTING_FOLDER, filename)
        index, name, distances = test_eigenfaces(test_path, model)

        # print the distances for this test image
        print("Test: %s" % filename)
        for i, d in enumerate(distances):
            marker = "  <-- minimum" if i == index else ""
            print("   %-18s %.4f%s" % (model["names"][i], d / 1e9, marker))
        print("   => recognised as %s\n" % name)

        # build the side-by-side figure
        test_face = np.asarray(plt.imread(test_path))
        recognised_face = model["images"][index]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(test_face, cmap="gray")
        axes[0].set_title("Test face")
        axes[0].axis("off")
        axes[1].imshow(recognised_face, cmap="gray")
        axes[1].set_title("Recognized face")
        axes[1].axis("off")
        fig.suptitle("%s  ->  %s" % (filename, name))
        plt.tight_layout()

        # save it into results/ with a matching name
        out_path = os.path.join(RESULTS_FOLDER, filename.replace(".pgm", ".png"))
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("   saved figure to %s\n" % out_path)


if __name__ == "__main__":
    main()