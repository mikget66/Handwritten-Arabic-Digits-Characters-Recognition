# Arabic Handwritten Digits Recognition

This project aims to recognize handwritten digits in the Arabic language using a convolutional neural network (CNN). It utilizes the Arabic Handwritten Digits Dataset available on Kaggle.

## Usage
1. Download the dataset from Kaggle: [Arabic Handwritten Digits Dataset](https://www.kaggle.com/datasets/mloey1/ahdd1).
2. Install the required dependencies: Matplotlib, Pandas, TensorFlow, Keras, PIL, numpy, and win32gui.
3. Update the file paths in the code to match the location of the downloaded dataset.
4. Run the code to train the model and save it as "model1.h5".
5. To use the GUI for digit recognition:
   - Run the code with the trained model file available.
   - Draw a digit on the canvas.
   - Click the "Recognise" button to classify the digit.

## Code Structure
- `Arabic Handwritten Digits Dataset CSV`: Directory containing the dataset CSV files.
- `train.py`: Script to train the CNN model on the dataset and save it.
- `gui.py`: Script implementing a GUI for digit recognition using the trained model.

## Dependencies
- Matplotlib
- Pandas
- TensorFlow
- Keras
- PIL
- numpy
- win32gui

## References
- [Arabic Handwritten Digits Dataset](https://www.kaggle.com/datasets/mloey1/ahdd1)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)


## Author
Michael Anwar

---
