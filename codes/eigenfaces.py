# Eigenfaces face recognition: training and testing functions

import glob
import os
import numpy as np
from PIL import Image


# read a grayscale / PGM image and return it as a float64 2D array
def load_image(path):
    return np.asarray(Image.open(path), dtype=np.float64)


# train the recogniser on every .pgm file inside the given folder
def train_eigenfaces(training_folder):
    # collect the training files, sorted so the order is always the same
    files = sorted(glob.glob(os.path.join(training_folder, "*.pgm")))
    if not files:
        raise FileNotFoundError("No .pgm files found in '%s'" % training_folder)

    images = [load_image(f) for f in files]            # original images (for display)
    names = [os.path.basename(f) for f in files]       # file names

    H, W = images[0].shape                             # height and width
    M = len(images)                                    # number of training images

    # turn each image into one long column vector -> matrix vec (H*W x M)
    vec = np.column_stack([img.reshape(-1) for img in images])

    # mean face: average of all training columns
    mean = vec.mean(axis=1, keepdims=True)             # H*W x 1

    # centre the data: subtract the mean face from every face
    A = vec - mean                                     # H*W x M

    # the real covariance A*A' would be 90000x90000, too large.
    # so we use the small matrix L = A'*A (M x M) and map the eigenvectors back with U = A*V
    L = A.T @ A                                         # M x M
    eigenvalues, V = np.linalg.eigh(L)                 # eigenvectors of L
    U = A @ V                                           # the eigenfaces, H*W x M

    # describe each training face by its projection onto the eigenfaces
    omega = U.T @ A                                     # M x M

    # pack everything testing needs into one dictionary
    model = {
        "images": images, "names": names,
        "H": H, "W": W, "M": M,
        "mean": mean, "U": U, "omega": omega,
    }
    return model


# recognise the face in test_image_path against a trained model
def test_eigenfaces(test_image_path, model):
    # read the test image and turn it into the same long column vector
    test = load_image(test_image_path).reshape(-1, 1)  # H*W x 1

    # centre it with the SAME mean face used during training
    centred = test - model["mean"]

    # project the test face onto the eigenfaces
    om = model["U"].T @ centred                        # M x 1

    # distance between the test projection and every training projection
    differences = om - model["omega"]                  # M x M
    distances = np.linalg.norm(differences, axis=0)    # one value per training face

    # the closest training face wins
    index = int(np.argmin(distances))
    return index, model["names"][index], distances