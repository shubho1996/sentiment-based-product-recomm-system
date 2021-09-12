# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:18:23 2021

@author: Shubhodeep Bhowmick
"""

from flask import Flask,request, render_template
from model import RecommendationSystem

model_object = RecommendationSystem()
app = Flask(__name__)

@app.route("/")
def home():
    """Renders home page of Flask app
    """
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    """Used to predict the top 5 recommended products
    """
    if(request.method == 'POST'):
        # Takes the username from the request and passes it to recommend product function
        username = request.form['text']
        product_list, is_correct_user = model_object.recommend_products(username)
        
        
        return render_template('results.html', products = product_list, 
                               is_correct_user = is_correct_user)
    


if __name__ == '__main__':
    app.run(debug=True)