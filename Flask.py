# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:11:37 2021

@author: Shubhodeep
"""

# Import dependencies
from flask import Flask


# Initialise Flask App
app = Flask(__name__)

# Define default route
@app.route("/")
def hello():
    return "Hello World!"
if __name__ == '__main__':
    app.run(debug=True)
