import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('D:\HealthcarePredictionApp\data\healthcare_dataset (1).csv')

# Feature selection
features = ['Age', 'Gender', 'BMI', 'Blood Pressure', 'Cholesterol (HDL/LDL)',
            'Glucose Level (mg/dL)', 'Family History', 'Existing Conditions']
target = 'Diagnosis Outcome'

# Preprocessing
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data['Family History'] = LabelEncoder().fit_transform(data['Family History'])
data['Existing Conditions'] = LabelEncoder().fit_transform(data['Existing Conditions'])

# Check if 'Blood Pressure' column contains strings before applying string operations
if data['Blood Pressure'].dtype == 'object': #If the column is of type object (string)
    data['Blood Pressure'] = data['Blood Pressure'].str.split('/').str[0].astype(int)  # Extract systolic BP
else:
    print("WARNING: 'Blood Pressure' column is not of string type. Skipping string operations.")
    # You may need to handle the existing numeric or mixed-type data appropriately.

# Check if 'Cholesterol (HDL/LDL)' column contains strings before applying string operations
if data['Cholesterol (HDL/LDL)'].dtype == 'object':  #If the column is of type object (string)
    data['Cholesterol'] = data['Cholesterol (HDL/LDL)'].str.split('/').str[1].astype(int) # Extract LDL Cholesterol
else:
    print("WARNING: 'Cholesterol (HDL/LDL)' column is not of string type. Skipping string operations.")
    # You may need to handle the existing numeric or mixed-type data appropriately.

# Update features to include 'Cholesterol' instead of 'Cholesterol (HDL/LDL)'
features = ['Age', 'Gender', 'BMI', 'Blood Pressure', 'Cholesterol', # Use the extracted 'Cholesterol' column
            'Glucose Level (mg/dL)', 'Family History', 'Existing Conditions'] 
target = 'Diagnosis Outcome'

X = data[features]
y = data[target]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 # Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
 
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Visualizations

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# 2. Gender Distribution (Pie Chart)
gender_counts = data['Gender'].value_counts()

# Get the unique gender values and their corresponding counts
gender_labels = gender_counts.index.tolist()  # Get labels from the index

# Map encoded values to original labels if needed
gender_labels = [('Male' if label == 1 else 'Female' if label == 0 else label) for label in gender_labels] 

#gender_labels = ['Male', 'Female', 'Other']  # Add a label for the third category if needed

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'pink', 'lightgreen']) # Added a color for the potential third category
plt.title("Gender Distribution")
plt.show()
# 3. Age Distribution (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(data['Age'], bins=15, kde=True, color='green')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
# 4. Diagnosis Outcome Count (Bar Plot)
outcome_counts = data['Diagnosis Outcome'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=outcome_counts.index, y=outcome_counts.values, palette="viridis")
plt.title("Diagnosis Outcome Count")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()


# 5 NEW Prediction Distribution (Bar Plot)
prediction_counts = pd.Series(y_pred).value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=prediction_counts.index, y=prediction_counts.values, palette="Blues")
plt.title("Prediction Outcome Distribution")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# 6 Actual vs Predicted (Grouped Bar Plot)
actual_vs_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_vs_pred_grouped = actual_vs_pred.groupby(['Actual', 'Predicted']).size().unstack().fillna(0)

plt.figure(figsize=(6, 4))
actual_vs_pred_grouped.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title("Actual vs Predicted Diagnosis Outcome")
plt.xlabel("Actual Outcome")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title='Predicted Outcome', labels=['No', 'Yes'])
plt.show()

# 7 Feature Importance using Coefficients from Logistic Regression (Bar Plot)
coefficients = model.coef_[0]  # Get coefficients of the logistic regression model
features_labels = features  # Features list remains the same as defined before

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': features_labels,
    'Importance': coefficients
})

# Sort the features by their importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette="viridis")
plt.title("Feature Importance in Predicting Diagnosis Outcome")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'model' and 'scaler' are already defined from the previous code

# Function to process the prediction
def health_prediction(user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, existing_conditions):
    # Gender encoding: Female -> 0, Male -> 1
    new_patient_data = np.array([[age, 1 if gender == "Male" else 0, bmi, blood_pressure, cholesterol,
                                 glucose_level, family_history, existing_conditions]])

    # Scale the new patient data using the same scaler used for training
    new_patient_data_scaled = scaler.transform(new_patient_data)

    # Make prediction using the trained model
    prediction = model.predict(new_patient_data_scaled)

    # Get prediction probability
    prediction_prob = model.predict_proba(new_patient_data_scaled)

    # Reasoning for prediction based on the inputs
    reasoning = []

    if age < 35:
        reasoning.append("Age: Lower risk due to younger age")
    else:
        reasoning.append("Age: Higher risk due to older age")

    if bmi < 25:
        reasoning.append("BMI: Normal weight, which is a healthy indicator")
    else:
        reasoning.append("BMI: Higher BMI, increasing risk")

    if blood_pressure < 120:
        reasoning.append("Blood Pressure: Healthy BP, lower risk")
    else:
        reasoning.append("Blood Pressure: High BP, increasing risk")

    if family_history == 0:
        reasoning.append("Family History: No genetic risk")
    else:
        reasoning.append("Family History: Increased genetic risk")

    if existing_conditions == 0:
        reasoning.append("Existing Conditions: No pre-existing conditions, lower risk")
    else:
        reasoning.append("Existing Conditions: Pre-existing conditions, increasing risk")

    # Prepare the result
    if prediction[0] == 'Yes':
        probability = prediction_prob[0][1] * 100  # Probability of 'Yes' (diagnosed)
        prediction_result = f"### Hello, **{user_name}**!\n"
        prediction_result += f"#### **Prediction**: **YES** (High likelihood of the disease)\n"
        prediction_result += f"#### Probability: {probability:.2f}%\n"
        prediction_result += "#### **Reasoning:**\n"
        for reason in reasoning:
            prediction_result += f"- {reason}\n"
        # Visualization: Display a pie chart for prediction probability
        fig, ax = plt.subplots()
        ax.pie([probability, 100 - probability], labels=["High Risk", "Low Risk"], autopct="%1.1f%%", startangle=90, colors=['#FF4C4C', '#4CAF50'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.close(fig)  # Close the plot to return it as an image
        return prediction_result, fig
    else:
        probability = prediction_prob[0][0] * 100  # Probability of 'No' (not diagnosed)
        prediction_result = f"### Hello, **{user_name}**!\n"
        prediction_result += f"#### **Prediction**: **NO** (Low risk)\n"
        prediction_result += f"#### Probability: {probability:.2f}%\n"
        prediction_result += "#### **Reasoning:**\n"
        for reason in reasoning:
            prediction_result += f"- {reason}\n"
        # Visualization: Display a pie chart for prediction probability
        fig, ax = plt.subplots()
        ax.pie([probability, 100 - probability], labels=["Low Risk", "High Risk"], autopct="%1.1f%%", startangle=90, colors=['#4CAF50', '#FF4C4C'])
        ax.axis('equal')
        plt.close(fig)  # Close the plot to return it as an image
        return prediction_result, fig


# Gradio UI
iface = gr.Interface(
    fn=health_prediction,
    inputs=[
        gr.Textbox(label="Enter your Name:"),
        gr.Number(label="Enter Age:", minimum=1, maximum=120, step=1),
        gr.Radio(["Female", "Male"], label="Select Gender:"),
        gr.Number(label="Enter BMI:", minimum=0.0, step=0.1),
        gr.Number(label="Enter Blood Pressure (Systolic):", minimum=50, maximum=200),
        gr.Number(label="Enter LDL Cholesterol:", minimum=50, maximum=300),
        gr.Number(label="Enter Glucose Level (mg/dL):", minimum=50.0, maximum=500.0),
        gr.Radio([0, 1], label="Family History (0 for No, 1 for Yes):"),
        gr.Radio([0, 1], label="Existing Conditions (0 for No, 1 for Yes):")
    ],
    outputs=[
        gr.Markdown(label="Prediction Result"),
        gr.Plot(label="Prediction Probability Pie Chart")
    ],
    title="Health Prediction Application",
    description="Welcome to the Health Prediction App! Enter your details below to get predictions. -created by  SHREYAS :)❤️ ",
)
# Launch Gradio UI
iface.launch()


'''
#FROM HERE REST OF THE CODE IN MY SQL INTERGRATED CODE- NOT FINISHED COMPLETELY#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mysql.connector
import numpy as np
import gradio as gr

# MySQL Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="dhanu@1009",  # your password
        database="health_prediction_db"
    )

# Load dataset
data = pd.read_csv('D:\HealthcarePredictionApp\data\healthcare_dataset (1).csv')

# Feature selection
features = ['Age', 'Gender', 'BMI', 'Blood Pressure', 'Cholesterol (HDL/LDL)',
            'Glucose Level (mg/dL)', 'Family History', 'Existing Conditions']
target = 'Diagnosis Outcome'

# Preprocessing
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data['Family History'] = LabelEncoder().fit_transform(data['Family History'])
data['Existing Conditions'] = LabelEncoder().fit_transform(data['Existing Conditions'])

# Check if 'Blood Pressure' column contains strings before applying string operations
if data['Blood Pressure'].dtype == 'object':
    data['Blood Pressure'] = data['Blood Pressure'].str.split('/').str[0].astype(int)
else:
    print("WARNING: 'Blood Pressure' column is not of string type. Skipping string operations.")

# Check if 'Cholesterol (HDL/LDL)' column contains strings before applying string operations
if data['Cholesterol (HDL/LDL)'].dtype == 'object':
    data['Cholesterol'] = data['Cholesterol (HDL/LDL)'].str.split('/').str[1].astype(int)
else:
    print("WARNING: 'Cholesterol (HDL/LDL)' column is not of string type. Skipping string operations.")

# Update features
features = ['Age', 'Gender', 'BMI', 'Blood Pressure', 'Cholesterol',
            'Glucose Level (mg/dL)', 'Family History', 'Existing Conditions'] 
target = 'Diagnosis Outcome'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization code omitted for brevity...

# Function to process the prediction
def health_prediction(user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, existing_conditions):
    # Gender encoding: Female -> 0, Male -> 1
    new_patient_data = np.array([[age, 1 if gender == "Male" else 0, bmi, blood_pressure, cholesterol,
                                 glucose_level, family_history, existing_conditions]])

    # Scale the new patient data using the same scaler used for training
    new_patient_data_scaled = scaler.transform(new_patient_data)

    # Make prediction using the trained model
    prediction = model.predict(new_patient_data_scaled)

    # Get prediction probability
    prediction_prob = model.predict_proba(new_patient_data_scaled)

    # Reasoning for prediction based on the inputs
    reasoning = []
    if age < 35:
        reasoning.append("Age: Lower risk due to younger age")
    else:
        reasoning.append("Age: Higher risk due to older age")

    if bmi < 25:
        reasoning.append("BMI: Normal weight, which is a healthy indicator")
    else:
        reasoning.append("BMI: Higher BMI, increasing risk")

    if blood_pressure < 120:
        reasoning.append("Blood Pressure: Healthy BP, lower risk")
    else:
        reasoning.append("Blood Pressure: High BP, increasing risk")

    if family_history == 0:
        reasoning.append("Family History: No genetic risk")
    else:
        reasoning.append("Family History: Increased genetic risk")

    if existing_conditions == 0:
        reasoning.append("Existing Conditions: No pre-existing conditions, lower risk")
    else:
        reasoning.append("Existing Conditions: Pre-existing conditions, increasing risk")

    # Prepare the result
    if prediction[0] == 'Yes':
        probability = prediction_prob[0][1] * 100  # Probability of 'Yes' (diagnosed)
        prediction_result = f"### Hello, **{user_name}**!\n"
        prediction_result += f"#### **Prediction**: **YES** (High likelihood of the disease)\n"
        prediction_result += f"#### Probability: {probability:.2f}%\n"
        prediction_result += "#### **Reasoning:**\n"
        for reason in reasoning:
            prediction_result += f"- {reason}\n"

        # Store the prediction results in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """INSERT INTO predictions (user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, 
                    family_history, existing_conditions, prediction, probability) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(query, (user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, 
                               existing_conditions, 'Yes', probability))
        conn.commit()
        cursor.close()
        conn.close()

        return prediction_result

    else:
        probability = prediction_prob[0][0] * 100  # Probability of 'No' (not diagnosed)
        prediction_result = f"### Hello, **{user_name}**!\n"
        prediction_result += f"#### **Prediction**: **NO** (Low risk)\n"
        prediction_result += f"#### Probability: {probability:.2f}%\n"
        prediction_result += "#### **Reasoning:**\n"
        for reason in reasoning:
            prediction_result += f"- {reason}\n"

        # Store the prediction results in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """INSERT INTO predictions (user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, 
                    family_history, existing_conditions, prediction, probability) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(query, (user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, 
                               existing_conditions, 'No', probability))
        conn.commit()
        cursor.close()
        conn.close()

        return prediction_result

# Gradio UI
iface = gr.Interface(
    fn=health_prediction,
    inputs=[
        gr.Textbox(label="Enter your Name:"),
        gr.Number(label="Enter Age:", minimum=1, maximum=120, step=1),
        gr.Radio(["Female", "Male"], label="Select Gender:"),
        gr.Number(label="Enter BMI:", minimum=0.0, step=0.1),
        gr.Number(label="Enter Blood Pressure (Systolic):", minimum=50, maximum=200),
        gr.Number(label="Enter LDL Cholesterol:", minimum=50, maximum=300),
        gr.Number(label="Enter Glucose Level (mg/dL):", minimum=50.0, maximum=500.0),
        gr.Radio([0, 1], label="Family History (0 for No, 1 for Yes):"),
        gr.Radio([0, 1], label="Existing Conditions (0 for No, 1 for Yes):")
    ],
    outputs=[
        gr.Markdown(label="Prediction Result"),
        gr.Plot(label="Prediction Probability Pie Chart")
    ],
    title="Health Prediction Application",
    description="Welcome to the Health Prediction App! Enter your details below to get predictions. -created by SHREYAS :)❤️ ",
)

# Launch Gradio UI
iface.launch()'''

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sqlite3
import numpy as np
import gradio as gr

# Load dataset
data = pd.read_csv('D:\\HealthcarePredictionApp\\data\\healthcare_dataset (1).csv')

# Handle missing data
data = data.dropna()  # Drop rows with missing values, or fill with mean

# Feature selection
features = ['Age', 'Gender', 'BMI', 'Blood Pressure', 'Cholesterol (HDL/LDL)',
            'Glucose Level (mg/dL)', 'Family History', 'Existing Conditions']
target = 'Diagnosis Outcome'

# Preprocessing
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data['Family History'] = LabelEncoder().fit_transform(data['Family History'])
data['Existing Conditions'] = LabelEncoder().fit_transform(data['Existing Conditions'])

# Handling 'Blood Pressure' values, which are in the form of '43/132' (e.g., systolic/diastolic)
if data['Blood Pressure'].dtype == 'object':
    # Split by '/' and take the first value (systolic)
    data['Blood Pressure'] = data['Blood Pressure'].str.split('/').str[0].astype(float)

# Handling 'Cholesterol (HDL/LDL)' values, which are in the form of 'HDL/LDL' (e.g., 45/120)
if data['Cholesterol (HDL/LDL)'].dtype == 'object':
    # Split by '/' and take the second value (LDL)
    data['Cholesterol'] = data['Cholesterol (HDL/LDL)'].str.split('/').str[1].astype(float)

# Redefine the features after cleaning
features = ['Age', 'Gender', 'BMI', 'Blood Pressure', 'Cholesterol', 'Glucose Level (mg/dL)', 'Family History', 'Existing Conditions']
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model with regularization
model = LogisticRegression(C=0.1, max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Database setup
conn = sqlite3.connect("health_predictions.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS Predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT,
    age INTEGER,
    gender TEXT,
    bmi REAL,
    blood_pressure INTEGER,
    cholesterol INTEGER,
    glucose_level REAL,
    family_history INTEGER,
    existing_conditions INTEGER,
    prediction TEXT,
    probability REAL
)
""")

# Function to save data to the database
def save_to_db(user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, existing_conditions, prediction, probability):
    cursor.execute("""
    INSERT INTO Predictions (user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, existing_conditions, prediction, probability)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, existing_conditions, prediction, probability))
    conn.commit()

# Gradio prediction function
def health_prediction(user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, existing_conditions):
    new_patient_data = np.array([[age, 1 if gender == "Male" else 0, bmi, blood_pressure, cholesterol,
                                  glucose_level, family_history, existing_conditions]])
    new_patient_data_scaled = scaler.transform(new_patient_data)
    
    prediction = model.predict(new_patient_data_scaled)
    prediction_prob = model.predict_proba(new_patient_data_scaled)

    reasoning = []
    if age < 35:
        reasoning.append("Age: Lower risk due to younger age")
    else:
        reasoning.append("Age: Higher risk due to older age")

    if bmi < 25:
        reasoning.append("BMI: Normal weight, which is a healthy indicator")
    else:
        reasoning.append("BMI: Higher BMI, increasing risk")

    if blood_pressure < 120:
        reasoning.append("Blood Pressure: Healthy BP, lower risk")
    else:
        reasoning.append("Blood Pressure: High BP, increasing risk")

    if family_history == 0:
        reasoning.append("Family History: No genetic risk")
    else:
        reasoning.append("Family History: Increased genetic risk")

    if existing_conditions == 0:
        reasoning.append("Existing Conditions: No pre-existing conditions, lower risk")
    else:
        reasoning.append("Existing Conditions: Pre-existing conditions, increasing risk")

    # Use correct probability index
    if prediction[0] == 1:
        probability = prediction_prob[0][1] * 100
    else:
        probability = prediction_prob[0][0] * 100

    save_to_db(user_name, age, gender, bmi, blood_pressure, cholesterol, glucose_level, family_history, existing_conditions, prediction[0], probability)

    prediction_result = f"### Hello, **{user_name}**!\n"
    prediction_result += f"#### **Prediction**: {'YES' if prediction[0] == 1 else 'NO'}\n"
    prediction_result += f"#### Probability: {probability:.2f}%\n"
    prediction_result += "#### **Reasoning:**\n"
    for reason in reasoning:
        prediction_result += f"- {reason}\n"

    return prediction_result

iface = gr.Interface(
    fn=health_prediction,
    inputs=[
        gr.Textbox(label="Enter your Name:"),
        gr.Number(label="Enter Age:", minimum=1, maximum=120, step=1),
        gr.Radio(["Female", "Male"], label="Select Gender:"),
        gr.Number(label="Enter BMI:", minimum=0.0, step=0.1),
        gr.Number(label="Enter Blood Pressure (Systolic):", minimum=50, maximum=200),
        gr.Number(label="Enter LDL Cholesterol:", minimum=50, maximum=300),
        gr.Number(label="Enter Glucose Level (mg/dL):", minimum=50.0, maximum=500.0),
        gr.Radio([0, 1], label="Family History (0 for No, 1 for Yes):"),
        gr.Radio([0, 1], label="Existing Conditions (0 for No, 1 for Yes):")
    ],
    outputs=[gr.Markdown(label="Prediction Result")],
    title="Health Prediction Application",
    description="Welcome to the Health Prediction App! Enter your details below to get predictions - created by SHREYAS :)❤️."
)

iface.launch()

'''



