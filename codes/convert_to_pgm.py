# Convert downloaded faces to 300x300 grayscale .pgm files

import os
from PIL import Image, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
# data folders may sit next to this script or one level up (e.g. in a codes/ folder)
ROOT = HERE if os.path.isdir(os.path.join(HERE, "for_training")) else os.path.dirname(HERE)
NEW_FACES = os.path.join(ROOT, "new_faces")
TRAINING_FOLDER = os.path.join(ROOT, "for_training")
TESTING_FOLDER = os.path.join(ROOT, "for_testing")

SIZE = 300  # target resolution, same as the other faces

# set these to your downloaded file names (inside the new_faces folder)
TRAIN_SOURCE = "myface.jpg"        # image that goes into training
TEST_SOURCE = "myface_test.jpg"    # a similar image for testing
# if you only downloaded ONE face, set TEST_SOURCE = None and a similar
# test image will be created automatically from the training one.


# center-crop to a square, convert to grayscale, resize to 300x300
def to_square_gray(path, size=SIZE):
    img = Image.open(path).convert("L")          # to grayscale
    w, h = img.size
    side = min(w, h)                             # crop to the largest centered square
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size))              # reduce to 300x300
    return img


# make a slightly different copy to act as a "similar" test photo (only used if TEST_SOURCE is None)
def make_similar(img):
    shifted = img.transform(img.size, Image.AFFINE, (1, 0, 6, 0, 1, 4))   # small shift
    shifted = ImageEnhance.Brightness(shifted).enhance(0.93)              # a bit darker
    return shifted


def main():
    os.makedirs(NEW_FACES, exist_ok=True)

    # training image -> for_training/myface.pgm
    train_img = to_square_gray(os.path.join(NEW_FACES, TRAIN_SOURCE))
    train_out = os.path.join(TRAINING_FOLDER, "myface.pgm")
    train_img.save(train_out)
    print("saved training face ->", train_out)

    # test image -> for_testing/myface_t.pgm
    if TEST_SOURCE:
        test_img = to_square_gray(os.path.join(NEW_FACES, TEST_SOURCE))
    else:
        test_img = make_similar(train_img)
        print("(no separate test photo given, created a similar one automatically)")
    test_out = os.path.join(TESTING_FOLDER, "myface_t.pgm")
    test_img.save(test_out)
    print("saved test face ->", test_out)


if __name__ == "__main__":
    main()