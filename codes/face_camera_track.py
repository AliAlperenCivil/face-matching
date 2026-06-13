# Real-time face detection, tracking and capture from the webcam

import os
import cv2
import numpy as np

from eigenfaces import train_eigenfaces, test_eigenfaces

HERE = os.path.dirname(os.path.abspath(__file__))
# data folders may sit next to this script or one level up (e.g. in a codes/ folder)
ROOT = HERE if os.path.isdir(os.path.join(HERE, "for_training")) else os.path.dirname(HERE)
TRAINING_FOLDER = os.path.join(ROOT, "for_training")
TESTING_FOLDER = os.path.join(ROOT, "for_testing")

SIZE = 300                 # captured faces are saved at 300x300, like the others
MODE = "train"             # "train" to add your face, "test" to recognise it
CAMERA_INDEX = 0           # 0 is usually the built-in webcam
MAX_MISSING = 15           # keep the box alive this many frames when detection blinks


# crop the box (a little enlarged), make it grayscale 300x300
def crop_face(gray_frame, box, size=SIZE):
    x, y, w, h = box
    pad = int(0.2 * w)
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, gray_frame.shape[1])
    y1 = min(y + h + pad, gray_frame.shape[0])
    face = gray_frame[y0:y1, x0:x1]
    face = cv2.resize(face, (size, size))
    return face


# blend the previous box with the new one to reduce jitter
def smooth_box(old, new, a=0.5):
    return tuple(int(a * o + (1 - a) * n) for o, n in zip(old, new))


def main():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print("Could not open the camera. Try changing CAMERA_INDEX to 1.")
        return

    prev_gray = None
    track_points = None
    last_box = None            # last good face box (kept for a few frames)
    missing = 0                # how many frames since we last saw a face
    captured_face = None

    print("MODE =", MODE, "| wait for green READY, press 'c' to capture, 'q' to quit")

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)                 # even out the lighting

        # detect faces (tuned to be more forgiving and stable)
        faces = face_detector.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))

        if len(faces) > 0:
            # pick the largest face (closest to the camera)
            box = max(faces, key=lambda b: b[2] * b[3])
            box = tuple(int(v) for v in box)
            last_box = smooth_box(last_box, box) if last_box is not None else box
            missing = 0

            # refresh corner points inside the face region
            x, y, w, h = last_box
            roi = gray[y:y + h, x:x + w]
            corners = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.01, minDistance=5)
            if corners is not None:
                corners = corners.reshape(-1, 2) + np.array([x, y], dtype=np.float32)
                track_points = corners.astype(np.float32)
        else:
            # no detection this frame: keep tracking the points and slide the old box along
            missing += 1
            if prev_gray is not None and track_points is not None and len(track_points) > 0:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, track_points, None)
                if new_points is not None:
                    good_new = new_points[status.flatten() == 1]
                    good_old = track_points[status.flatten() == 1]
                    if len(good_new) > 0 and last_box is not None:
                        shift = np.median(good_new - good_old, axis=0)   # how far the face moved
                        x, y, w, h = last_box
                        last_box = (int(x + shift[0]), int(y + shift[1]), w, h)
                    track_points = good_new
            if missing > MAX_MISSING:
                last_box = None                          # really lost the face now

        # decide whether we currently have a usable face
        have_face = last_box is not None and missing <= MAX_MISSING

        if have_face:
            x, y, w, h = last_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            if track_points is not None:
                for px, py in track_points:
                    cv2.drawMarker(frame, (int(px), int(py)), (255, 255, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
            cv2.putText(frame, "READY - press 'c' to capture", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "searching for a face...", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.putText(frame, "MODE: %s   ('q' quit)" % MODE, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.imshow("Face detect and track", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and have_face:
            captured_face = crop_face(gray, last_box)
            break

        prev_gray = gray.copy()

    cam.release()
    cv2.destroyAllWindows()

    if captured_face is None:
        print("No face captured.")
        return

    if MODE == "train":
        out = os.path.join(TRAINING_FOLDER, "camera_face.pgm")
        cv2.imwrite(out, captured_face)
        print("Saved training face ->", out)
        print("Now run run_all_tests.py once, then use MODE = 'test'.")
    else:
        out = os.path.join(TESTING_FOLDER, "camera_test.pgm")
        cv2.imwrite(out, captured_face)
        print("Saved test face ->", out)

        model = train_eigenfaces(TRAINING_FOLDER)
        index, name, distances = test_eigenfaces(out, model)
        print("\ndist (x 1e9):")
        for i, d in enumerate(distances):
            marker = "  <-- minimum" if i == index else ""
            print("  %-18s %.4f%s" % (model["names"][i], d / 1e9, marker))
        print("\nRecognised as:", name)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(captured_face, cmap="gray")
        axes[0].set_title("Test face (camera)")
        axes[0].axis("off")
        axes[1].imshow(model["images"][index], cmap="gray")
        axes[1].set_title("Recognized face")
        axes[1].axis("off")
        fig.suptitle("camera_test.pgm  ->  %s" % name)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()