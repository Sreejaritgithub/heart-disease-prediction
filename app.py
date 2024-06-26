from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Predict', methods=['POST'])
def prediction():
    # Get form data
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    
    # Create input array for prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Predict
    prediction = model.predict(input_data)
    
    # Determine output message
    if prediction[0] == 1:
        output = 'Heart Disease'
    else:
        output = 'No Heart Disease'
    
    # Render the template with prediction result
    return render_template('index.html', prediction_text='Predicted healthiness of heart is : {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
