# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:51:00 2019

@author: tallurim
"""
import numpy as np
from flask import Flask, request, jsonify, render_template #
import pickle

app = Flask(__name__) #Creating flask app
model = pickle.load(open('model.pkl', 'rb')) #Loading model from pickle

#Home Page..by /
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)