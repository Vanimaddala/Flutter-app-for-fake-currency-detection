from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import base64
from PIL import Image
import io
import cv2
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import kurtosis, skew, entropy
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv('banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']

# Balance the dataset
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
target_count = data.auth.value_counts()
nb_to_delete = target_count[0] - target_count[1]
data = data[nb_to_delete:]

# Prepare training and testing data
x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the model
clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
clf.fit(x_train, y_train.values.ravel())

# Evaluate the model
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Helper function to convert base64 string to image
def string_to_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return image

# Helper function to convert image to edge-detected version
def string_to_edge_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    img_blur = cv2.GaussianBlur(np.array(image), (3, 3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    return np.array(sobelxy)

# Route to handle file uploads and processing
@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            uploaded_file = request.files['image']

            if uploaded_file.filename == '':
                return jsonify({"error": "No selected file."})

            if not uploaded_file.mimetype.startswith('image/'):
                return jsonify({"error": "Invalid file type. Please upload an image file."})

            # Save the uploaded image
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Process the uploaded image
            opencv_image = cv2.cvtColor(np.array(Image.open(file_path)), cv2.COLOR_RGB2BGR)
            norm_image = np.array(opencv_image, dtype=np.float32) / 255.0

            # Extract features
            var = np.var(norm_image, axis=None)
            sk = skew(norm_image, axis=None)
            kur = kurtosis(norm_image, axis=None)
            ent = entropy(norm_image, axis=None) / 100

            # Predict using the trained model
            result = clf.predict(np.array([[var, sk, kur, ent]]))
            authenticity = "Real Currency" if result[0] == 0 else "Fake Currency"

            # Prepare the response data
            response_data = {
                "result": authenticity,
                "variance": f"{var:.2f}",
                "skewness": f"{sk:.2f}",
                "kurtosis": f"{kur:.2f}",
                "entropy": f"{ent:.2f}",
            }

            return jsonify(response_data)

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
