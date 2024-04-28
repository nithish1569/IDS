import zipfile
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import h2o
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)

# Initialize H2O
h2o.init()

# Load the saved H2O model
model_path = r"C:\Users\nithi\OneDrive\Desktop\TEAM\saved_model\DRF_model_python_1713967822737_1"
loaded_model = h2o.load_model(model_path)

# Define paths for saving datasets and results
data_folder = r"C:\Users\nithi\OneDrive\Desktop\TEAM\dataset"
train_data_path = os.path.join(data_folder, "train.csv")
test_data_path = os.path.join(data_folder, "test.csv")
results_folder = r"C:\Users\nithi\OneDrive\Desktop\TEAM\results"

# Email configuration
email_address = "palle.nithishreddy@gmail.com"
email_password = "ydju kzte rqqs xscn"

# Function to split and save dataset
def split_and_save_dataset(data_path, train_path, test_path, test_size=0.2):
    # Read dataset
    df = pd.read_csv(data_path)
    
    # Split dataset into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Save train and test datasets
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

# Function to make predictions and save results as CSV
def make_predictions_and_save_results(test_df):
    # Convert test dataset to H2OFrame
    test_df_hex = h2o.H2OFrame(test_df)
    
    # Make predictions using the loaded model
    predictions = loaded_model.predict(test_df_hex)
    
    # Convert predictions to pandas DataFrame
    predictions_df = predictions.as_data_frame()
    
    # Combine predictions with original test dataset
    results_df = pd.concat([test_df, predictions_df], axis=1)
    
    # Generate unique filename based on current timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file_path = os.path.join(results_folder, f"prediction_results_{timestamp_str}.csv")
    
    # Save results as CSV file
    results_df.to_csv(results_file_path, index=False)

    return results_file_path  # Return the path to the saved CSV file

# Function to send email with attachment
def send_email_with_attachment(receiver_email, subject, body, file_path):
    sender_email = email_address
    sender_password = email_password
    
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    
    # Add body to email
    message.attach(MIMEText(body, "plain"))
    
    # Compress the file
    compressed_file_path = compress_file(file_path)
    
    # Open compressed file in binary mode
    with open(compressed_file_path, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    
    # Encode file in ASCII characters to send via email
    encoders.encode_base64(part)
    
    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {os.path.basename(compressed_file_path)}",
    )
    
    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()
    
    # Log in to server and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, text)

# Function to compress file
def compress_file(file_path):
    zip_file_path = file_path + '.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path), compress_type=zipfile.ZIP_DEFLATED)
    return zip_file_path

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']
    if username == "palle.nithishreddy@gmail.com" and password == "6309562806@Ns":
        return redirect(url_for('upload_image'))
    else:
        return "Invalid credentials. Please try again."

@app.route('/upload_image')
def upload_image():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['file']
        
        # Save uploaded file
        uploaded_file_path = os.path.join(data_folder, file.filename)
        file.save(uploaded_file_path)
        
        # Split and save dataset
        split_and_save_dataset(uploaded_file_path, train_data_path, test_data_path)
        
        # Read test dataset
        test_df = pd.read_csv(test_data_path)
        
        # Make predictions and save results as CSV
        results_file_path = make_predictions_and_save_results(test_df)
        
        # Send email with attachment
        send_email_with_attachment("palle.nithishreddy@gmail.com", "Prediction Results", "Please find the prediction results attached.", results_file_path)
        
        # Read prediction results
        prediction_results = pd.read_csv(results_file_path).to_dict(orient='records')
        
        # Prepare message to display on result.html
        message = f"Prediction results saved at: {results_file_path}. Also sent to palle.nithishreddy@gmail.com"
        
        return render_template('result.html', message=message, prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
