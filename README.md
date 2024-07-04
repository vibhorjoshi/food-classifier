Food Classification App
Project Overview
This project is a Streamlit-based web application that uses a pre-trained deep learning model to classify food images. Users can upload an image of food, and the app will predict the type of food in the image.
Project Link
[Insert your project repository link here]
Tech Stack

Python 3.x
Streamlit
TensorFlow / Keras
Pandas
NumPy
Pillow (PIL)

Features

User-friendly web interface for image upload
Real-time food classification
Displays top prediction and confidence score
Shows top 3 predictions with their probabilities
Supports JPG and JPEG image formats

How to Run the Project
Prerequisites

Python 3.x installed
pip (Python package installer)

Installation

Clone the repository:
Copygit clone [Your repository URL]
cd [Your project directory]

Create and activate a virtual environment (optional but recommended):
Copypython -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

Install the required packages:
Copypip install -r requirements.txt


Running the App

Ensure you're in the project directory and your virtual environment is activated (if you're using one).
Run the Streamlit app:
Copystreamlit run app.py

Open a web browser and go to the URL provided by Streamlit (usually http://localhost:8501).

Usage

Once the app is running, you'll see an option to upload an image.
Click on "Choose an image.." and select a JPG or JPEG image of food.
The app will display the uploaded image and provide a prediction of the food type.
You'll see the top prediction along with the top 3 predictions and their confidence scores.

Model Information

The app uses a pre-trained deep learning model for food classification.
The model file should be named final_food_classifier.h5 and placed in the project root directory.
Class names for predictions are read from a file named class_names.csv.

Troubleshooting

If you encounter a warning about tf.reset_default_graph being deprecated, you can safely ignore it as it's coming from an internal Keras function.
Ensure all required files (final_food_classifier.h5 and class_names.csv) are present in the project directory.

Contributing
[Insert information about how others can contribute to your project]
License
[Insert your license information here]
