from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

application = Flask(__name__)   
app = application

regressor_model = pickle.load(open("models/regressor.pkl","rb"))
standard_scaler = pickle.load(open("models/scaler.pkl","rb"))


## Route for home page 
@app.route("/predictdata")
def index():
    return render_template("index.html")

@app.route("/", methods=["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        MedInc = request.form.get("MedInc")
        HouseAge = request.form.get("HouseAge")
        AveRooms = request.form.get("AveRooms")
        AveBedrms = request.form.get("AveBedrms")
        Population = request.form.get("Population")
        AveOccup = request.form.get("AveOccup")
        Latitude = request.form.get("Latitude")
        Longitude = request.form.get("Longitude")

        new_data_scaled = standard_scaler.transform([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
        result = regressor_model.predict(new_data_scaled)

        return render_template("home.html",result=result[0])

    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
