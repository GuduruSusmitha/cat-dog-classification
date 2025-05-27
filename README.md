import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
# Load the trained model
# Assuming the model is saved directly in your Google Drive's MyDrive folder
# Check this path carefully! Make sure the file 'dog and cat_cnn_model.h5' exists here.
try:
    # Corrected the path to point to the actual model file
    model = tf.keras.models.load_model('/content/drive/MyDrive/dog and cat_cnn_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check the model path and file format.")
    # You might want to add a fallback or exit here if the model fails to load

# Preprocessing function
def predict_tumor(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Ensure the model prediction output shape is as expected
    # The original code assumes a single output neuron for binary classification
    prediction = model.predict(img_array)
    # Check if the prediction has the expected shape (e.g., (1, 1))
    if prediction.shape == (1, 1):
        prediction_value = prediction[0][0]
    elif prediction.shape == (1, 2): # Example for a model with two output neurons (softmax)
        # Assuming the model outputs probabilities for two classes (cat, dog)
        # Adjust indexing based on your model's output order
        prediction_value = prediction[0][1] # Example: probability of being 'dog'
    else:
        # Handle unexpected output shape
        print(f"Warning: Unexpected model prediction shape: {prediction.shape}")
        # Fallback or raise an error
        prediction_value = 0 # Defaulting to cat

    return "dog" if prediction_value > 0.5 else "cat"

# Gradio Interface
interface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(label="Prediction"),
    title="dog and cat Detector",
    description="Upload an MRI scan image to check for brain tumor (dog/cat)"
)

interface.launch()
