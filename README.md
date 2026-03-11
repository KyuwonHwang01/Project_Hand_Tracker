## Hand Gesture Sign Translator

This project uses MediaPipe hand landmarks to:

- record labeled hand-sign samples into `data/<LABEL>/*.json`
- compare live hand motion against saved samples
- append stable predictions into an on-screen translated text buffer

### Requirements

- Python 3.10+
- `opencv-python`
- `mediapipe`

### Run

```bash
python3 main.py
```

### Controls

- `r`: start recording a sample for a label such as `A` or `HELLO`
- `s`: stop recording and save the sample, then reload templates
- `c`: clear translated text
- `Backspace`: delete the last translated character
- `q`: quit

### How recognition works

The recognizer loads saved JSON recordings from `data/`, normalizes the landmark positions for left and right hands, samples each recording into a fixed-length sequence, and finds the closest live match with a distance threshold.

This is a baseline template-matching system. For better real-world sign language accuracy, the next step is usually a trained sequence model using many samples per sign.
