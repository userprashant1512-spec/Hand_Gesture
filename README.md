
# Hand Gesture Recognition using OpenCV and Machine Learning

## Project Overview
This project recognizes hand gestures in real time using a webcam. It uses OpenCV for video capture, MediaPipe for hand landmark detection, and a Random Forest classifier trained on collected hand landmark data.

## Features
- Real-time webcam hand detection
- Recognition of five hand gestures
- Custom dataset collection
- Random Forest model training
- Tkinter GUI dashboard
- Mouse and presentation controls
- Emergency stop for computer controls

## Gestures
| Label | Gesture |
|---|---|
| 0 | Open Palm |
| 1 | Fist |
| 2 | Peace |
| 3 | Thumbs Up |
| 4 | One Finger |

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- PyAutoGUI
- Tkinter

## Project Files
- `main.py` — main GUI application
- `collect_data.py` — collects hand landmark samples
- `train_model.py` — trains the Random Forest classifier
- `test_trained_model.py` — tests the trained model
- `gesture_model.pkl` — saved trained model
- `requirements.txt` — Python dependencies

## Installation
1. Install Python 3.12.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Create and activate a virtual environment.
5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Start the GUI application:

```bash
python main.py
```

Test the trained model:

```bash
python test_trained_model.py
```

## Dataset and Model
The model is trained using 63 hand landmark features per sample (x, y, z coordinates for 21 landmarks). The training script saves the model as `gesture_model.pkl`.

## Limitations
Performance depends on lighting, camera quality, hand position, and the training dataset. The reported test accuracy may overestimate real-world performance because adjacent webcam frames can be similar.

## Future Improvements
- Improve performance on unseen hand positions
- Add more gestures
- Improve mouse control and gesture stability
- Develop a browser-based version

## Author
B.Tech AIML Student
