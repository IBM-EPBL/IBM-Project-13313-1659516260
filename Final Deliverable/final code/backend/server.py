import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for 
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
model = tf.keras.models.load_model("vegetable.h5")
model1 = tf.keras.models.load_model("fruit.h5")

recommendation = {"apple_rotten":"Before fertilizing apple trees, know your boundaries. Mature trees have large root systems that can extend outwards 1 ½ times the diameter of the canopy and can be 4 feet (1 m.) deep. These deep roots absorb water and store excess nutrients for the successive year, but there are also smaller feeder roots that reside in the top foot (30.5 cm.) of soil that absorb most nutrients",
                  "Corn_blight": "For sweetcorn plants look for a fertilizer that contains an N-P-K ratio with a higher proportion of nitrogen this is because sweetcorn plants will quickly deplete the soil of this important macronutrient. In addition, your growing sweetcorn plant will also need phosphorus as well as potassium until it reaches maturity. For example, a sweetcorn fertilizer with the numbers 20-15-15 on the label means that this product contains 20% nitrogen, 15% phosphorous, and 15% potassium.",
                  "Peach_Bacteria":"Mature peach trees mostly require nitrogen (N) and potassium (K), the two nutrients found at higher concentrations in fruits. Phosphorus encourages root development and is essential for young trees. Use a complete fertilizer, such as 16-4-8, 12-6-6, 12-4-8, or 10-10-10, during the tree's first three years",
                  "Pepper_bacteria":" An ideal fertilizer ratio for fruiting tomatoes, peppers, and eggplants is 5-10-10 with trace amounts of magnesium and calcium added.",
                  "Potato_blight":"For a hardy crop, you want a fertilizer NPK of around 2-2-3 (2% Nitrogen – 2% Phosphorus – 3% Potassium). You’ll also need the means to balance out your soil’s acidity.  Adding compost to your beds will really help to introduce additional nutrients that your potatoes crop will love. Plus, I have a few tricks up my sleeve to introduce some specific nutrients that target and boost individual stages of growth.",
                  "Tomato_Bacteria":"The best proportion of fertilizers for tomato is 5-10-10",
                  "Tomato_Leafmold":"The best proportion of fertilizers for tomato is 5-10-10",
                  "Tomato_blight":"The best proportion of fertilizers for tomato is 5-10-10",
}

fruit_labels = ['apple_rotten','apple_healthy','corn_healthy','Corn_blight','Peach_Bacteria','Peach_Healthy']
vegetable_labels = ['Pepper_bacteria','Pepper_healthy','Potato_blight',"Potato_healthy","Potato_blight",'Tomato_Bacteria',"Tomato_blight","Tomato_Leafmold",'Tomato_Leafmold']
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template('predict.html')

Solution = ""
label=""

@app.route('/prediction',methods=['POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img = tf.keras.preprocessing.image.load_img(file_path,target_size =(128,128))
        x= tf.keras.utils.img_to_array(img)
        x = x.reshape((1,128,128,3))
        plant = request.form['plant']
        if(plant=='vegetable'):
            preds = model.predict(x)
            label = vegetable_labels[np.argmax(preds[0])]
            Solution = recommendation[label]
             
        else:
            preds = model1.predict(x)
            label = fruit_labels[np.argmax(preds[0])]
            Solution = recommendation[label]

        return render_template('prediction.html',recommendation=Solution,Disease=label)

if __name__ == "__main__":
    app.run(debug=True)