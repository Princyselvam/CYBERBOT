# CYBERBOT
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:52:19 2025

@author: princ
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:00:09 2024

@author: princ
"""

import streamlit as st
from twilio.rest import Client
import os
import cv2
import numpy as np
import pickle
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import bcrypt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from twilio.rest import Client
import socket
import sys
import requests
import re
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import joblib


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# Define the file to store user data
USER_DATA_FILE = 'user_data.csv'


# Define the file to store user data
USER_DATA_FILE = 'user_data.csv'
if not os.path.isfile(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        f.write('username,password\n')

# Function to register users
def register_user(username, password):
    # Check if username already exists
    user_data = pd.read_csv(USER_DATA_FILE)
    if username in user_data['username'].values:
        st.warning("Username already exists. Please choose a different username.")
        return
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    with open(USER_DATA_FILE, 'a') as f:
        f.write(f'{username},{hashed_password.decode("utf-8")}\n')
    st.success("Registration successful! You can now log in.")

# Function to check login credentials and update session state
def check_login(username, password):
    user_data = pd.read_csv(USER_DATA_FILE)
    user = user_data[user_data['username'] == username]
    if not user.empty:
        stored_password = user.iloc[0]['password']
        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            return True
    return False


# Function to register face data
def register_face(name):
    if not name:
        st.warning("Please enter a name to register face data.")
        return
    
    st.info("Starting face data registration. Please look at the camera.")
    
    # Start capturing video from webcam
    video = cv2.VideoCapture(0)
    mugam_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mugam_data = []
    i = 0
    
    # Capture loop
    while True:
        ret, frame = video.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in mugam:
            # Crop and resize face area
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))
            
            # Append face data if conditions are met
            if len(mugam_data) < 100 and i % 10 == 0:
                mugam_data.append(resized_img)
            
            i += 1
            
            # Display number of captured faces and draw rectangle
            cv2.putText(frame, str(len(mugam_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
        # Show frame with rectangle and counter
        cv2.imshow("Frame", frame)
        
        # Break loop if 'q' is pressed or 100 faces are captured
        k = cv2.waitKey(1)
        if k == ord('q') or len(mugam_data) == 100:
            break
    
    # Release video capture and close all windows
    video.release()
    cv2.destroyAllWindows()

    # Convert the face data to numpy array and reshape
    mugam_data = np.asarray(mugam_data)
    mugam_data = mugam_data.reshape(100, -1)

    # Save name data in 'names.pkl'
    if 'names.pkl' not in os.listdir("."):
        names = [name] * 100
        with open("names.pkl", 'wb') as f:
            pickle.dump(names, f)
    else:
        with open("names.pkl", 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * 100
        with open("names.pkl", 'wb') as f:
            pickle.dump(names, f)

    # Save face data in 'mugam_data.pkl'
    if 'mugam_data.pkl' not in os.listdir("."):
        with open("mugam_data.pkl", 'wb') as f:
            pickle.dump(mugam_data, f)
    else:
        with open("mugam_data.pkl", 'rb') as f:
            mugam = pickle.load(f)
        mugam = np.append(mugam, mugam_data, axis=0)
        with open("mugam_data.pkl", 'wb') as f:
            pickle.dump(mugam, f)
    
    st.success(f"Face data for {name} has been registered successfully.")
# Ensure OpenCV uses a compatible backend on Windows
import os
os.environ["QT_QPA_PLATFORM"] = "windows"

# Function to display images using Matplotlib (fix for Spyder issues)
def show_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis("off")
    plt.show()

# Function to take attendance using face recognition
def take_attendance():
    # Ensure face data exists
    if not os.path.exists('names.pkl') or not os.path.exists('mugam_data.pkl'):
        print("❌ No face data found. Please register your face first.")
        return
    
    print("📷 Starting attendance. Please look at the camera...")

    # Load face recognition model and data
    mugam_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    video = cv2.VideoCapture(0)  # Try 1 instead of 0 if the camera doesn't open
    
    if not video.isOpened():
        print("❌ Error: Unable to access the camera.")
        return

    with open('names.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('mugam_data.pkl', 'rb') as f:
        MUGAM = pickle.load(f)

    # Ensure feature and label data have the same length
    if len(MUGAM) != len(labels):
        min_length = min(len(MUGAM), len(labels))
        MUGAM = MUGAM[:min_length]  # Trim excess
        labels = labels[:min_length]  # Trim excess
        print(f"⚠️ Warning: Data mismatch fixed! Now using {min_length} samples.")

    # Train the model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(MUGAM, labels)

    # Prepare attendance file
    col = ['Name', "Time"]
    attendance_taken = set()
    
    detected_name = "Unknown"

    while True:
        ret, frame = video.read()
        if not ret:
            print("❌ Error: Failed to access the webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in mugam:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            detected_name = output[0]

            # Draw rectangle and label around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Mark attendance only if not already marked
            if detected_name not in attendance_taken:
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                attendance = [detected_name, timestamp]

                csv_file = f"attendance_{date}.csv"
                excel_file = f"attendance_{date}.xlsx"
                file_exists = os.path.isfile(csv_file)

                # Save attendance to CSV
                with open(csv_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(col)  # Write headers if file doesn't exist
                    writer.writerow(attendance)

                # Save attendance to Excel
                if os.path.exists(excel_file):
                    df = pd.read_excel(excel_file)
                    new_entry = pd.DataFrame([attendance], columns=col)
                    df = pd.concat([df, new_entry], ignore_index=True)
                else:
                    df = pd.DataFrame([attendance], columns=col)

                df.to_excel(excel_file, index=False)

                attendance_taken.add(detected_name)
                print(f"✅ Attendance marked for {detected_name} at {timestamp}.")

        # Show frame using OpenCV (Use Matplotlib in Spyder if needed)
        try:
            cv2.imshow("Attendance Monitoring", frame)
        except cv2.error:
            show_image(frame)  # Use Matplotlib instead

        if cv2.waitKey(1) & 0xFF == ord('q') or len(attendance_taken) > 0:
            break

    video.release()
    cv2.destroyAllWindows()



# Function to preprocess text for chatbot
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Chatbot function
def run_chatbot():
    st.subheader("Chatbot")
    
    if not os.path.exists('question_response_pairs.pickle'):
        st.warning("Chatbot data not found. Please ensure 'question_response_pairs.pickle' exists.")
        return

    with open('question_response_pairs.pickle', 'rb') as file:
        question_response_pairs = pickle.load(file)

    user_input = st.text_input("You: ")
    if user_input:
        user_tokens = preprocess_text(user_input)
        similarities = {}
        for question, response in question_response_pairs.items():
            question_tokens = preprocess_text(question)
            intersection = set(user_tokens) & set(question_tokens)
            union = set(user_tokens) | set(question_tokens)
            jaccard_similarity = len(intersection) / len(union) if len(union) != 0 else 0
            similarities[question] = jaccard_similarity
        best_question = max(similarities, key=similarities.get)
        response = question_response_pairs.get(best_question, "I'm not sure how to respond to that.")
        st.write(f"Bot: {response}")
# Sample dataset
data = {
    'url': [
        'http://example.com', 'https://secure-site.com', 'http://malicious-site.com',
        'https://safe-browsing.com', 'http://dangerous-url.net', 'http://www.google.com',
        'https://www.microsoft.com/en-us', 'https://github.com/login',
        'http://www.stackoverflow.com/questions', 'https://docs.python.org/3/', 'http://www.amazon.in/deals',
        'https://www.linkedin.com/in/sample', 'http://developer.mozilla.org/en-US/', 'https://www.apple.com/in/',
        'http://www.wikipedia.org', 'https://www.khanacademy.org', 'http://www.airbnb.com/home',
        'https://web.whatsapp.com', 'http://zoom.us/signin', 'https://www.netflix.com/browse',
        'http://news.bbc.co.uk', 'https://www.indianrail.gov.in', 'http://www.icicibank.com',
        'https://maps.google.com', 'http://www.timesofindia.indiatimes.com', 'https://www.flipkart.com',
        'http://www.nytimes.com', 'https://www.tesla.com', 'https://www.zoho.com', 'http://www.researchgate.net',

        # Malicious URLs
        'http://verify-paypal-login.com', 'https://www.login-facebook.com.verify-user.net',
        'http://www.bankofamerica.secure-login-alert.com', 'http://free-giftcards.www.winnerclub.in',
        'https://www.netflix-account-verification.com', 'http://update.appleid.confirm-reset.com',
        'https://secure.amazon-login-check.com', 'http://www.freefirehacktool.com/claim',
        'https://www.get-rich-fast.click', 'http://bit.ly/fake-login-page', 'https://dropbox-login-security-alert.com',
        'http://www.stealbankdetails.com/update', 'https://win-vouchers-quick.com', 'http://newyearbonus.www.fakeoffer.in',
        'http://secure-mail-recovery.ru', 'https://account-unusual-activity.com/login',
        'http://www.resetpass.google.mail-support.com', 'https://www.winiphone13free.com',
        'http://malicious-downloads.com/file.exe', 'http://bank.login-verify-alert.com',
        'https://login-update-amazon.secure.com', 'https://www-facebook-login-alert.com/verify',
        'http://paypal.account.alerts-update.com', 'https://update-your-id-apple.com.login.verify',
        'http://getyourprize-now.win-bonus.click', 'https://dangerous-free-prizes.fakeoffer.com'
    ],
    'label': [
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1,   # Good URLs end here (index 0–29)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1               # Malicious URLs (index 30–55)
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

df = pd.DataFrame(data)
# Feature extraction
def extract_features(url):
    return [
        len(url),
        url.count('.'),
        url.count('/'),
        url.count('-'),
        1 if url.startswith('https') else 0,
        1 if re.match(r'\d+\.\d+\.\d+\.\d+', url) else 0
    ]

X = np.array([extract_features(url) for url in df['url']])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, 'url_classifier.pkl')
st.write("Enter a URL to check if it is Benign or Malicious")

# Password Hashing
def hash_menu():
    st.subheader("Password Hashing")
    password = st.text_input("Password", type="password")
    if password:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        st.write("Hashed Password:", hashed.decode('utf-8'))
# Load Twilio credentials from environment variables (to keep them secure)
ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
HR_PHONE_NUMBER = os.getenv('HR_PHONE_NUMBER')

# Load trained model and vectorizer
try:
    clf = joblib.load("email_classifier.pkl")  # Trained model
    vectorizer = joblib.load("vectorizer.pkl")  # Corresponding vectorizer
except Exception as e:
    st.error(f"🚨 Error loading model/vectorizer: {e}")
    st.stop()  # Stop execution if model files are missing

# Function to classify emails
def classify_email(file):
    """Processes uploaded email file and predicts if it's phishing or normal."""
    try:
        email_text = file.read().decode("utf-8")  # Read email content
        email_features = vectorizer.transform([email_text])  # Extract features
        prediction = clf.predict(email_features)[0]  # Predict
        
        # Return classification result
        return "Normal" if prediction == 0 else "Phishing"
    except Exception as e:
        return f"Error: {e}"


# Check if Twilio API is reachable before proceeding
def check_network():
    try:
        ip = socket.gethostbyname("api.twilio.com")
        print(f"Twilio API is reachable at {ip}")
    except socket.gaierror:
        print("Error: Cannot resolve Twilio API. Check your network settings.")
        sys.exit(1)

# Initialize Twilio client
def get_twilio_client():
    try:
        return Client(ACCOUNT_SID, AUTH_TOKEN)
    except Exception as e:
        print(f"Error initializing Twilio client: {e}")
        sys.exit(1)

# Function to send an emergency SMS
def send_emergency_sms(client, user_id, message):
    try:
        body = f"Emergency call from user {user_id}: {message}"
        sms = client.messages.create(
            body=body,
            from_=TWILIO_PHONE_NUMBER,
            to=HR_PHONE_NUMBER
        )
        return sms.sid
    except Exception as e:
        print(f"Failed to send emergency SMS: {e}")
        sys.exit(1)

# Function to place an emergency phone call
def place_emergency_call(client, user_id):
    try:
        call = client.calls.create(
            twiml=f'<Response><Say>Emergency call from user {user_id}. Please respond immediately.</Say></Response>',
            from_=TWILIO_PHONE_NUMBER,
            to=HR_PHONE_NUMBER
        )
        return call.sid
    except Exception as e:
        print(f"Failed to place emergency call: {e}")
        sys.exit(1)


def main_menu():
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""

    # Show login or the rest of the app based on login status
    if not st.session_state['logged_in']:
        menu = ["Login", "Register"]
    else:
        menu = ["Home", "Register Face", "Take Attendance", "Chatbot", "URL Classification", "Password Hashing","Email classification", "Emergency Call", "Logout"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(f"Welcome, {username}!")
            else:
                st.warning("Invalid username or password.")
    
    elif choice == "Register":
        st.subheader("Register New Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if password == confirm_password and username:
                register_user(username, password)
                st.success("Registration successful! You can now log in.")
            else:
                st.warning("Passwords do not match or username is missing.")
    
    elif choice == "Home" and st.session_state['logged_in']:
        st.subheader(f"Welcome {st.session_state['username']}!")
        st.write("You are now logged in. Choose a feature from the sidebar.")
    
    elif choice == "Register Face" and st.session_state['logged_in']:
        name = st.text_input("Enter your name:")
        if st.button("Register Face"):
            register_face(name)
            st.success("Face registered successfully!")
    
    elif choice == "Take Attendance" and st.session_state['logged_in']:
        st.subheader("Face Recognition for Attendance")
        if st.button("Take Attendance"):
            take_attendance()
            st.success("Attendance taken successfully!")
    
    elif choice == "Chatbot" and st.session_state['logged_in']:
        run_chatbot()
    
    elif choice == "extract features" and st.session_state.get('logged_in', False):
        extract_features()

    # Load the trained model (ensure the model file exists)
    clf = joblib.load('url_classifier.pkl')

    # Input field for URL
    url_input = st.text_input("Enter a URL to analyze:", key="url_input")

    # Button for extracting features and classifying
    if st.button("Analyze URL"):
        if url_input.strip():  # Check if input is not empty
            # Extract features
            url_features = np.array(extract_features(url_input)).reshape(1, -1)

            # Classify the URL
            prediction = clf.predict(url_features)[0]

            # Display results
            if prediction == 0:
                st.success("This URL is classified as **Benign** ✅")
            else:
                st.error("This URL is classified as **Malicious** ❌")

        else:
            st.warning("Please enter a URL to analyze.")



    elif choice == "Password Hashing" and st.session_state['logged_in']:
        hash_menu()
    
    elif choice == "classify email":
        st.subheader("📧 Email Phishing Classifier")
        st.write("Upload an email file to check if it's **Normal** or **Phishing**.")

    uploaded_file = st.file_uploader("📂 Upload an email file (TXT, EML)", type=["txt", "eml"])

    if uploaded_file:
        st.text_area("📜 Email Content Preview:", uploaded_file.getvalue().decode("utf-8"), height=200)

        classification_result = classify_email(uploaded_file)

        # Display result
        st.subheader("🧐 Classification Result:")
        if classification_result == "Normal":
            st.success("✅ This email is classified as **Normal**.")
        elif classification_result == "Phishing":
            st.error("❌ This email is classified as **Phishing**!")
        else:
            st.warning(f"⚠️ {classification_result}")  # Show error message if any

    elif choice == "Emergency Call" and st.session_state['logged_in']:
        st.subheader("Emergency Call")
        client = get_twilio_client()
        user_id = st.session_state['username']
        message = st.text_area("Enter emergency message (optional):")
        
        if st.button("Send Emergency SMS"):
            sid = send_emergency_sms(client, user_id, message)
            st.success(f"Emergency SMS sent successfully! SID: {sid}")
        
        if st.button("Place Emergency Call"):
            sid = place_emergency_call(client, user_id)
            st.success(f"Emergency call placed successfully! SID: {sid}")
    
    elif choice == "Logout" and st.session_state['logged_in']:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.success("Logged out successfully!")

if __name__ == "__main__":
    st.title("Cyberbot Application")
    main_menu()
