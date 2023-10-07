from flask import Flask, render_template,request
import pandas as pd
import pickle
import os

model_path = os.path.abspath("RidgeModel.pkl")

app=Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')

pipe = pickle.load(open(model_path,'rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    availabilities=sorted(data['availability'].unique())
    areas=sorted(data['area_type'].unique())
    return render_template('index1.html', location=locations,availability=availabilities,area_type=areas)

@app.route('/predict', methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('sqft')
    availability=request.form.get('availability')
    area_type=request.form.get('area_type')

    print(location, bhk, bath, sqft,availability,area_type)
    input=pd.DataFrame([[location, sqft, bath, bhk,availability,area_type]],columns=['location', 'total_sqft', 'bath', 'bhk','availability','area_type'])
    prediction = pipe.predict(input)[0] * 100000

    return str(round(prediction,2))

if __name__=="__main__":
    app.run(debug=True , port=5001)