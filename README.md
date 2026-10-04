# AMC

A Python script that uses MediaPipe hand tracking to control the mouse cursor and clicks via hand gestures.

## Motivation
Provides a hands‑free way to move the mouse and perform clicks using webcam‑detected hand gestures.

## Tech stack
- Python
- MediaPipe
- OpenCV
- pynput
- tkinter
- winsound / playsound for audio feedback

## Features
- Real‑time hand detection and tracking
- Cursor movement mapped to hand position
- Left‑click and right‑click gestures
- Optional floating status dot UI
- Configurable mouse speed, smoothing, dead‑zone, and gesture confirmation frames
- Audio feedback for click events

## Installation
```sh
git clone https://github.com/pratham-jain33/AMC.git
cd AMC
python -m venv venv
source venv/bin/activate   # on Windows use `venv\\Scripts\\activate`
pip install -r requirements.txt
```

## Usage
```sh
python main.py
```
Edit the configuration variables at the top of `main.py` (e.g., `SHOW_UI`, `MOUSE_SPEED`, `ENABLE_DOT_UI`) to customize behavior.

## Build status
No continuous integration configured. To verify the project, install the dependencies and run `python main.py`; the script should start the webcam feed and allow mouse control.

## Code style
The code follows standard Python conventions: 4‑space indentation, snake_case naming, double quotes for strings, and imports grouped by standard library, third‑party, and local modules.

## Code example
```python
# Enable the on‑screen UI dot
SHOW_UI = True
ENABLE_DOT_UI = True
```

## API reference
The repository provides a single executable script `main.py`; it does not expose a reusable library API.

## Tests
No test suite is included in the repository.

---

*Created with [repo-doctor](https://prathamjain.com/projects/repo-doctor)*
