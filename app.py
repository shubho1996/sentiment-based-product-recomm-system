# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:18:23 2021

@author: Shubhodeep Bhowmick
"""

from flask import Flask,request, render_template
import joblib


df_reviews = joblib.load(r'models/processed_reviews_df.pkl')
sentiment_model = joblib.load(r'models/sentiment_model.pkl')
user_ratings = joblib.load(r'models/user_ratings.pkl')
vectorizer = joblib.load(r'models/vectorizer.pkl')
user_dict = joblib.load(r'models/user_dictionary.pkl')
product_dict = joblib.load(r'models/product_dictionary.pkl')
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
        product_list, is_correct_user = recommend_products(username)
        
        
        return render_template('results.html', products = product_list, 
                               is_correct_user = is_correct_user)
    

def recommend_products(username):
    """This function takes username as input and gives top 5 recommend items.

    Args:
        username (text): Username of the user in dataset

    Returns:
        tuple: Product list, is username correct or not
    """
    # Check whether username is proper or not
    if username not in user_dict:
        return ("", False)
    
    # get the user_id for the user name associated
    # Recommending the Top 20 products to the user.
    product_id_list = user_ratings.loc[user_dict[username]].sort_values(ascending=False)[0:20]
    
    # Convert product id list to product names
    product_names = []
    for i in product_id_list.index:
        product_names.append(product_dict[i])
        
    # Pass the above list to sentiment model and find the 5 top most products
    product_positive_scores = []
    for product in product_names:
        # Find all the relevant reviews for the specific product
        product_reviews = df_reviews[df_reviews.name == product].processed_review
        
        #Convert the reviews to numerical features
        X_values = vectorizer.transform(product_reviews)
        
        # Find the average positivity rate for the product
        positive_sum = 0
        probs = sentiment_model.predict_proba(X_values)
        for x in probs:
            positive_sum += x[1]
        product_positive_scores.append(positive_sum/len(probs))
    
    product_positive_zip = zip(product_names, product_positive_scores)
    
    # Sort based on positivity rate
    product_positive_zip_sorted = sorted(product_positive_zip, key=lambda x:x[1], reverse=True)
    final_product_list, _ =zip(*product_positive_zip_sorted)
    return (final_product_list[:5],True)


if __name__ == '__main__':
    app.run(debug=True)