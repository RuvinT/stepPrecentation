#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 01:47:44 2023

@author: ruvinjagoda
"""

import mysql.connector
from flask import Flask, jsonify

app = Flask(__name__)

# Connection to database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Ruvin@123",
  database="tempdatabase"
)

mycursor = mydb.cursor()

# Get request send to get graph data from the database
@app.route("/stock_prices", methods=["GET"])
def stock_prices():
    # SQL query to select all columns from stock_prices table
    mycursor.execute("SELECT * FROM stock_prices")
    result = mycursor.fetchall()
    stocks = [{"x": date, "open": open,"high":high,"low":low,"close":close,"volume": volume} 
              for date, open, high, low, close, volume in result]
    response = jsonify(stocks)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    app.run(debug=True)