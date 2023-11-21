import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)


# Load the pre-trained model
model = load_model('model.h5')
    
@app.route('/')
def start():
    return render_template("start.html")

@app.route("/end")
def end():
    return render_template("start.html")

@app.route('/result', methods=["POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        labels = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']
        img = image.load_img(filepath, target_size=(128, 128))
        x = image.img_to_array(img)
        x = x / 255
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # Make predictions using the loaded model
        prediction = model.predict(img_data)
        predicted_class = np.argmax(prediction, axis=1)
        x = list(prediction.keys())
        y = list(prediction.values())
        return pred, {x: y for x, y in zip(labels, prediction)}

        return render_template('predictions.html', prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
