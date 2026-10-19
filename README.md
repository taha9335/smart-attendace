# Smart Attendance

A Python and Flask prototype for webcam face recognition and daily attendance tracking. OpenCV detects faces, a locally supplied model predicts labels, and attendance records are stored in CSV files.

## Current status

This repository contains the backend and face detector, but it is **not yet a complete runnable demo**. The application references `templates/home.html` and `static/face_recognition_model.pkl`; neither is included in the current checkout. A registration and model-training workflow is also not implemented in `app.py`.

## Implemented behavior

- Stream webcam frames through `/video_feed` using the server's camera.
- Detect faces with the bundled Haar cascade.
- Load a local recognition model when available.
- Record a recognized user's name, ID, and time in a daily CSV.
- Avoid duplicate attendance entries for the same user ID on the same day.

## Stack

Python, Flask, OpenCV, NumPy, pandas, and joblib.

## Development setup

1. Clone this repository and open its root directory.
2. Create a Python virtual environment and install `requirements.txt`.
3. Supply `templates/home.html` and a compatible, trusted recognition model at `static/face_recognition_model.pkl`.
4. Ensure the machine running Flask has a working webcam.
5. Run `python app.py` and open `http://127.0.0.1:1000/`.

The model must accept flattened 50 × 50 colour face crops and return labels in `Name_ID` format. The app creates its attendance and face-data directories automatically.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Flask routes, video streaming, recognition, and CSV recording |
| `haarcascade_frontalface_default.xml` | Face detection cascade |
| `confusionmatrix.py` | Evaluation helper script |
| `requirements.txt` | Python dependency list |

## Limitations and next steps

The app tracks one detected label in shared process state, so it is not designed for independent simultaneous camera sessions. Complete the missing UI and training workflow before presenting it as an end-to-end demo. Authentication, configuration of the Flask secret, and deployment hardening remain future work; debug mode is enabled in the current entry point.

Use consented sample data for demonstrations and keep face images, trained identity models, and attendance records out of public commits.

## Contributing

Contributions that complete the setup, improve the interface, or document reproducible evaluation are welcome. Describe the change and how it was checked in your pull request.
