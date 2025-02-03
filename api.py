import csv
from quart import Quart, jsonify, request, send_from_directory
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import socket
import datetime
import pickle
import calendar
import joblib
import io
import time
import os
import signal
import threading
import subprocess

app = Quart(__name__)

ip_started = {}
csv_file_path = 'mluzon.csv'
modelName = "dbn_model.h5"
model = load_model(modelName)

features = ['Month_April', 'Month_August', 'Month_December', 'Month_February', 
            'Month_January', 'Month_July', 'Month_June', 'Month_March', 
            'Month_May', 'Month_November', 'Month_October', 'Month_September',
            'Weekday or Weekend_Weekday', 'Weekday or Weekend_Weekend', 
            'Type of Day_Normal Day', 'Type of Day_Regular Holiday', 
            'Type of Day_Special Non-working Holiday',
            'Mean Temperature (Degree Celsius)', 'Rainfall(mm)', 
            'Relative Humidity (%)', 'Windspeed (m/s)']

with open('scaler_dbn_X.pkl', 'rb') as f:
    scaler_dbn_X = joblib.load(f)
with open('scaler_dbn_y.pkl', 'rb') as f:
    scaler_dbn_y = joblib.load(f)


def load_trained_model(file_path):
    if file_path != "dbn_model.pkl":
        return load_model(file_path)
    else:
        return joblib.load(file_path)


def predict_demand_load(model, modelName, sequence_input):
    prediction_scaled = model.predict(sequence_input)
    scalerY = getScalerY(modelName)
    if len(prediction_scaled.shape) == 3 and modelName == "dbn_model.h5":
        prediction_scaled = prediction_scaled[:, -1, :]
    prediction = scalerY.inverse_transform(prediction_scaled)
    print("")
    return prediction[0][0]


def getScalerX(input_data):
    return preprocess_input(scaler_dbn_X, input_data)


def getScalerY(modelName):
    return scaler_dbn_y


async def open_predict_model(desired_month, desired_day, desired_year, month, day, weekdayOrWeekEnd, typeOfDay, temperature, rainfall, humidity, wind_speed):
  
    addition = 1.5
    count = 0.1

    date = f"{desired_month}/{desired_day}/{desired_year}"
    input_data = {}
    input_data['Date'] = date
    input_data = getMonthValue(input_data,month)
    input_data[month] = True
    input_data[weekdayOrWeekEnd] = True
    input_data[typeOfDay] = True
    input_data['Mean Temperature (Degree Celsius)'] = temperature
    input_data['Rainfall(mm)'] = rainfall
    input_data['Relative Humidity (%)'] = humidity
    input_data['Windspeed (m/s)'] = wind_speed
    print(f'\n\n\n{input_data}\n\n')
    result = {}
    X_scaled = getScalerX(input_data)

    time_steps = 7
    if len(X_scaled) < time_steps:
        X_scaled = np.tile(X_scaled, (time_steps, 1))
    X_seq = np.array([X_scaled])

    prediction = predict_demand_load(model, modelName, X_seq)

    print(f"Predicted Demand Load {date}:", prediction)
    result[str(date)] = prediction
    count += 1
    return result

def getMonthValue(input_data, month):
    # List of all months from January to December
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    
    # Loop through each month and set the value in input_data
    for m in months:
        input_data["Month_" + m] = (m == month)

    return input_data

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"Error getting local IP: {e}")
        return None


def preprocess_input(scaler_X, input_data):
    # Convert input data to DataFrame with an index
    df = pd.DataFrame([input_data])

    # One-hot encode categorical features
    # df = pd.get_dummies(df, columns=['day', 'weekdayOrWeekEnd', 'typeOfDay'])

    # Add missing columns with value 0
    for col in features:
        if col not in df.columns:
            df[col] = False

    # Reorder columns to match the training data format
    df = df[features]

    # Normalize the features
    X_scaled = scaler_X.transform(df.values)

    return X_scaled



def validateWeekday(week):
    if (week.lower() == "weekday" or week.lower() == "weekend"):
        return "Weekday or Weekend_"+week
    else:
        return jsonify({"message": "weekdayOrWeekEnd must be either Weekday or Weekend"}), 400


def getMonth(month):
    if (month == 1):
        return "Month_January"
    elif (month == 2):
        return "Month_February"
    elif (month == 3):
        return "Month_March"
    elif (month == 4):
        return "Month_April"
    elif (month == 5):
        return "Month_May"
    elif (month == 6):
        return "Month_June"
    elif (month == 7):
        return "Month_July"
    elif (month == 8):
        return "Month_August"
    elif (month == 9):
        return "Month_September"
    elif (month == 10):
        return "Month_October"
    elif (month == 11):
        return "Month_November"
    elif (month == 12):
        return "Month_December"
    else:
        return jsonify({"message": "month must be either this values: 1,2,3,4,5,6,7,8,9,10,11,12"}), 400


def getTypeOfDay(typeOfDay):
    if (typeOfDay == "Regular Holiday" or typeOfDay == "Special Non-working Holiday" or typeOfDay == "Normal Day"):
        return "Type of Day_"+typeOfDay
    else:
        return jsonify({"message": "typeOfDay must be either this values: Normal Day, Regular Holiday, Special Non-working Holiday"}), 400


def validateDay(day):
    if (day.lower() == "monday" or day.lower() == "tuesday" or day.lower() == "wednesday" or day.lower() == "thursday" or day.lower() == "friday" or day.lower() == "saturday" or day.lower() == "sunday"):
        return "Day_"+day
    else:
        return jsonify({"message": "day must be either of this values: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday"}), 400


@app.route("/predict", methods=['POST'])
async def predict():
    # Get the uploaded CSV file
    mFile = await request.files

    if 'file' not in mFile:
        return jsonify({"message": "No file part in the request"}), 400

    file = mFile['file']
    if file.filename == '':
        return jsonify({"message": "No file selected"}), 400

    # Read CSV data from the file
    try:
        # Read the CSV data into a DataFrame
        data = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
    except Exception as e:
        return jsonify({"message": "Error reading CSV file", "error": str(e)}), 400

    # Check if DataFrame is empty
    if data.empty:
        return jsonify({"message": "CSV file is empty"}), 400
    
    if len(data) > 100:
        return jsonify({"message": "CSV file contains more than 100 records. Only 100 records are allowed."}), 400

    results = []
    for _, row in data.iterrows():
        date_value = row['Date']  # Assume this is "12/31/2024"
        desired_month, desired_day, desired_year = date_value.split('/')
        month = getMonth(int(desired_month))
        # day = validateDay(row['day'])
        day = ""
        weekdayOrWeekEnd = validateWeekday(row['weekdayOrWeekEnd'])
        typeOfDay = getTypeOfDay(row['typeOfDay'].strip())
        temperature = row['temperature']
        rainfall = row['rainfall']
        humidity = row['humidity']
        wind_speed = row['wind_speed']

        result = await open_predict_model(desired_month, desired_day, desired_year, month, day, weekdayOrWeekEnd, typeOfDay, temperature, rainfall, humidity, wind_speed)
        if result:
            for date, prediction in result.items():
                result[date] = float(prediction)
            results.append(result)
            time.sleep(1)
        else:
            return jsonify({"status": "failed"}), 500
    return jsonify(results), 200


@app.route("/manual/predict", methods=['POST'])
async def manualPredict():
    # Get the uploaded CSV file
    mFile = await request.files

    if 'file' not in mFile:
        return jsonify({"message": "No file part in the request"}), 400

    file = mFile['file']
    if file.filename == '':
        return jsonify({"message": "No file selected"}), 400

    # Read CSV data from the file
    try:
        # Read the CSV data into a DataFrame
        data = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
    except Exception as e:
        return jsonify({"message": "Error reading CSV file", "error": str(e)}), 400

    # Check if DataFrame is empty
    if data.empty:
        return jsonify({"message": "CSV file is empty"}), 400

    results = []
    for _, row in data.iterrows():
        print(row)
        try:
            date_value = row['Date']  # Assume this is "12/31/2024"
            desired_month, desired_day, desired_year = date_value.split('/')
        except Exception as e:
            print(e)
            return jsonify({"status": "failed"}), 500

        month = getMonth(int(desired_month))
        day = ""
        weekdayOrWeekEnd = validateWeekday(row['Weekday or Weekend'])
        typeOfDay = getTypeOfDay(row['Type Of Day'].strip())
        temperature = row['Mean Temperature']
        rainfall = row['Rainfall']
        humidity = row['Relative Humidity']
        wind_speed = row['Windspeed']
        print([desired_month, desired_day, desired_year, month, day,
              weekdayOrWeekEnd, typeOfDay, temperature, rainfall, humidity, wind_speed])

        result = await open_predict_model(desired_month, desired_day, desired_year, month, day, weekdayOrWeekEnd, typeOfDay, temperature, rainfall, humidity, wind_speed)
        if result:
            for date, prediction in result.items():
                result[date] = float(prediction)
            results.append(result)
            time.sleep(1)
        else:
            return jsonify({"status": "failed"}), 500
    return jsonify(results), 200


@app.route("/train/new/model", methods=['POST'])
async def train():
    return


@app.route("/health", methods=['GET'])
def serve_home():
    return jsonify({"status": "OK"}), 200


def restart_server():
    time.sleep(2)
    command = f"start cmd /k start_api.bat"
    subprocess.Popen(command, shell=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Restart the Flask development server
    os.kill(os.getpid(), signal.SIGINT)

if __name__ == "__main__":

    # local_ip = get_local_ip()
    app.run(host="0.0.0.0", debug=True, port=5000, use_reloader=False)
