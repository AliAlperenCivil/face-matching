# Face Recognition with Eigenfaces

A small face recognition project for the *Image Processing & Recognition* course.
It implements the **Eigenfaces** method (PCA applied to face images) in Python and is a
re-implementation of the MATLAB scripts `trainingEF.m` and `testingEF.m`. It also adds a
real-time webcam mode for face detection, tracking and capture.

## How it works

Each `300x300` grayscale image is treated as one point in a 90,000-dimensional space (one
dimension per pixel). The steps are:

1. Compute the **mean face** of the training set and subtract it from every face (centering).
2. Find the main directions in which faces differ from the average. These directions are the
   **eigenfaces**. Because the real covariance matrix would be `90000 x 90000`, we use the
   small matrix `L = Aᵀ·A` (size `M x M`) and map its eigenvectors back with `U = A·V`.
3. Describe each training face by its **projection** onto the eigenfaces.
4. To recognise a new face, project it the same way and pick the training face whose
   projection is closest (smallest Euclidean distance).

## Project structure

```
.
├── eigenfaces.py          # core: train_eigenfaces() and test_eigenfaces()
├── run_recognition.py     # train + recognise a single test image
├── run_all_tests.py       # train once + test several images, saves figures to results/
├── convert_to_pgm.py      # turn downloaded faces into 300x300 grayscale .pgm
├── face_camera_track.py   # real-time webcam detection, tracking and capture
├── requirements.txt
├── for_training/          # training faces (.pgm)
├── for_testing/           # test faces (.pgm)
├── new_faces/             # put your downloaded images here (for convert_to_pgm.py)
└── results/               # saved result figures
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Single test**

```bash
python run_recognition.py
```
Trains on `for_training/`, recognises one image from `for_testing/` (edit `TEST_IMAGE` in the
file), prints the distance to every training face and shows the test face next to the match.

**Batch test**

```bash
python run_all_tests.py
```
Trains once and tests every image listed in `TEST_IMAGES`. A `Test face / Recognized face`
figure is saved into `results/` for each one.

**Add your own face from an AI generator**

1. Download a face (e.g. from generated.photos) and a similar one for testing.
2. Put them in `new_faces/`.
3. Set `TRAIN_SOURCE` and `TEST_SOURCE` at the top of `convert_to_pgm.py`
   (or set `TEST_SOURCE = None` to create a similar test image automatically from one photo).
4. Run it — the faces are converted to `300x300` grayscale `.pgm` and placed in the
   training and testing folders.

```bash
python convert_to_pgm.py
```

**Real-time camera**

```bash
python face_camera_track.py
```
1. With `MODE = "train"`, run it. Wait for the green **READY** text, press `c` to capture
   (or `q` to quit). Your face is saved into `for_training/`.
2. Run `run_all_tests.py` once so training includes the new face.
3. Set `MODE = "test"` and run again. Capture your face once more with `c`. It is saved into
   `for_testing/`, recognised against the training set, and a result figure is shown.

Tips: use good front lighting, face the camera directly, and stay close enough that the face
is at least ~80 px. If the camera does not open, change `CAMERA_INDEX` to `1`.

## Notes

The distances are unnormalised, so their absolute size is not meaningful on its own; only the
*relative* order matters (the smallest distance wins). Pixel ordering during flattening does
not affect the result as long as it is consistent between training and testing.
