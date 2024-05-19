# Import required libraries
import face_recognition as fr  # Necessary for face recognition
import numpy as np  # Necessary for numerical operations
from threading import Thread  # Necessary for threading
import cv2  # Necessary for computer vision tasks
from flask import Flask, request, render_template, flash, url_for, redirect, session, jsonify  # Necessary for Flask web app
from werkzeug.security import check_password_hash, generate_password_hash  # Necessary for password hashing
from firebase_admin import credentials, initialize_app, db, storage  # Necessary for Firebase operations
import hashlib, sys, firebase_admin # Necessary for various functionalities
import secrets, requests, base64, os  # Necessary for various functionalities
from datetime import datetime, timedelta # Necessary for date/time operations
import time
from flask_session import Session
import pandas as pd
#_________________________________________________________________________________________________________
app = Flask(__name__, template_folder='template')
config = {"databaseURL": "https://flaskwebapp-e3844-default-rtdb.firebaseio.com",
          "storageBucket": "flaskwebapp-e3844.appspot.com"}

secret_key = secrets.token_hex(24)
app.secret_key = secret_key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

if not firebase_admin._apps:
    cred = credentials.Certificate("credentials.json")
    initialize_app(cred, config)
#______________________________________________________________________________________________________

# Declare known_names and known_name_encodings initially as an empty dictionary
known_face_names = []
known_face_encodings = []

# Example: Load known faces from Firebase Storage
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # Reference to the Firebase Storage bucket
    bucket = storage.bucket()
    try:
        # List all files in the bucket
        blobs = bucket.list_blobs()

        for blob in blobs:
            # Get voter ID from the filename (assuming the filename is in the format "voterid.jpg")
            voter_id = os.path.splitext(blob.name)[0]

            # Read the image from the Firebase Storage blob
            image_bytes = blob.download_as_bytes()
            image_np = cv2.imdecode(np.asarray(bytearray(image_bytes), dtype="uint8"), cv2.IMREAD_COLOR)

            # Extract face encoding for the first face in the image
            encoding = fr.face_encodings(image_np)[0]

            # Append the encoding and corresponding name to the lists
            known_face_encodings.append(encoding)
            known_face_names.append(voter_id.capitalize())  # Use the voter ID as the name

    except Exception as e:
        print(f"Error loading known faces: {e}")

    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

# Function to hash passwords
def hash_password(password):
    hash_algorithm = hashlib.sha256()
    password_bytes = password.encode('utf-8')
    hash_algorithm.update(password_bytes)
    hashed_password = hash_algorithm.hexdigest()
    return hashed_password

# Function to save base64 image to Firebase Storage and get the public URL
def save_base64_image_to_firebase(data_url, file_name):
    _, encoded = data_url.split(",", 1)
    decoded_image = base64.b64decode(encoded)

    # Upload the image to Firebase Storage
    blob = storage.bucket().blob(file_name)
    blob.upload_from_string(decoded_image, content_type='image/jpg')

    # Get the public URL of the uploaded image
    photo_url = blob.public_url
    return photo_url

########################################____REGISTER_ROUTE____################################################

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        voter_id = request.form['voter_id']
        password = request.form['password']

        # Hash the password during registration
        hashed_password = generate_password_hash(password)
        print("Hashed Password:", hashed_password)

        captured_photo = request.form['captured_photo']

        # Save the photo directly to Firebase Storage with the voter ID as part of the filename
        photo_filename = f"{voter_id}.jpg"
        photo_url = save_base64_image_to_firebase(captured_photo, photo_filename)

        # Save the registration information to the database using voter_id as the key
        user_ref = db.reference(f'users/{voter_id}')
        user_ref.set({
            'name': name,
            'age': age,
            'voter_id': voter_id,
            'password_hash': hashed_password,
            'photo_url': photo_url
        })
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

#________________________________________________________________________________________________________
# Function to train the face recognition model
def train_model(user_image_bytes, voter_id, known_face_encodings, known_face_names):
    try:
        # Convert the image bytes to a NumPy array
        image_np = cv2.imdecode(np.frombuffer(user_image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Extract face encoding for the first face in the image
        encoding = fr.face_encodings(image_np)[0]

        # Update the known_face_encodings list with the new encoding only for the current user
        if voter_id.capitalize() in known_face_names:
            index = known_face_names.index(voter_id.capitalize())
            known_face_encodings[index] = encoding
        else:
            known_face_encodings.append(encoding)
            known_face_names.append(voter_id.capitalize())  # Use the voter ID as the name

        #flash('Face recognition model trained successfully!', 'success')

    except Exception as e:
        print(f"Error training model for user {voter_id}: {e}")
        flash('Error training face recognition model. Please try again.', 'danger')

#########################################____LOGIN_ROUTE____#####################################################

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        voter_id = request.form['voter_id']
        password = request.form['password']

        users_ref = db.reference('users')
        users_snapshot = users_ref.order_by_child('voter_id').equal_to(voter_id).get()

        user = next(iter(users_snapshot.values()), None)
        print("User:", user)

        if user:
            stored_password_hash = user.get('password_hash')
            print("Stored Password Hash:", stored_password_hash)

            if stored_password_hash:
                if check_password_hash(stored_password_hash, password):
                    print("Password Matched!")

                    # Load the user's image from Firebase Storage using their voter ID
                    user_image_url = f"{voter_id}.jpg"
                    user_image_bytes = load_image_from_firebase(user_image_url)

                    # Load known faces for the current user
                    user_known_face_names, user_known_face_encodings = load_known_faces_for_user(voter_id)

                    # Train the face recognition model with the user's image
                    train_model(user_image_bytes, voter_id, user_known_face_encodings, user_known_face_names)

                    # Flash success message
                    session['user_id'] = voter_id
                    # Redirect to the detect_face route with the user_id
                    flash("Login successful!")
                    return redirect(url_for('home', user_id=voter_id))
                else:
                    print("Invalid password entered during login.")
                    flash("Invalid password. Please try again.")
            else:
                print("Stored password hash is missing.")
                flash("Invalid password hash. Please contact support.")
        else:
            print("User not found.")
            flash("User not found. Please check your voter ID.")

    # If the request method is GET or any error occurred during login, show the login page
    return render_template('login.html')

########################################___HOME.HTML___#############################################################################

@app.route('/home/<user_id>')
def home(user_id):
    user_ref = db.reference(f'users/{user_id}')
    user_data = user_ref.get()

    if user_data:
        # Get user details
        user_name = user_data.get('name', 'Unknown')
        user_age = user_data.get('age', 'Unknown')
        voter_id = user_data.get('voter_id', 'Unknown')

        # Construct the user's image URL based on voter_id
        user_image_url = f"https://firebasestorage.googleapis.com/v0/b/{config['storageBucket']}/o/{voter_id}.jpg?alt=media"

        return render_template('home.html', user_name=user_name, user_age=user_age,
                               voter_id=voter_id, user_image_url=user_image_url)
    else:
        return redirect(url_for('home', user_id=user_id))

####################################____FACE_RECOGNITION_MODEL____###########################################

# Function to load image from Firebase Storage
def load_image_from_firebase(image_url):
    blob = storage.bucket().blob(image_url)

    # Read the image from the Firebase Storage blob
    image_bytes = blob.download_as_bytes()
    return image_bytes

# Function to load known faces for a specific user
def load_known_faces_for_user(logged_in_user_id):
    global known_face_encodings, known_face_names

    user_known_face_encodings = []
    user_known_face_names = []

    for name, encoding in zip(known_face_names, known_face_encodings):
        if name.lower() == logged_in_user_id.lower():
            user_known_face_names.append(name)
            user_known_face_encodings.append(encoding)

    return user_known_face_names, user_known_face_encodings


def detect_and_recognize_face(video_capture, user_id, known_face_encodings):
    face_recognized = False  # Flag to track if a face is recognized
    match_found = False  # Flag to track if a match is found

    start_time = time.time()  # Record the start time

    # Create the OpenCV window
    cv2.namedWindow("Detecting Face", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Detecting Face", 400, 0)  # Adjust the position as needed
    cv2.resizeWindow("Detecting Face", 540, 380)  # Set the initial size of the window

    while True:
        # Read a frame from the video capture object
        ret, frame = video_capture.read()
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Find face locations and encodings in the current frame
        face_locations = fr.face_locations(frame)
        face_encodings = fr.face_encodings(frame, face_locations)

        # Check if at least one face is detected
        if len(face_encodings) > 0:
            # Consider only the first detected face
            face_encoding = face_encodings[0]

            # Check if known_face_encodings is not empty
            if len(known_face_encodings) > 0:
                # Compare face encodings with known face encodings for the logged-in user only
                for known_encodings in known_face_encodings:
                    # Check if known_encodings is not empty
                    if len(known_encodings) > 0:
                        # Convert known_encodings to a NumPy array
                        known_encodings_np = np.array(known_encodings)

                        # Expand dimensions if needed
                        if len(known_encodings_np.shape) == 1:
                            known_encodings_np = np.expand_dims(known_encodings_np, axis=0)

                        # Compare face encodings using np.linalg.norm
                        distances = np.linalg.norm(known_encodings_np - face_encoding, axis=1)

                        # Set the face_recognized flag since a face is detected
                        face_recognized = True

                        # If any distance is less than a threshold (you can set a threshold here),
                        # consider it a match
                        if any(distances <= 0.5):  # Adjust the threshold as needed
                            match_found = True

            # Display a rectangle around the first detected face
            (top, right, bottom, left) = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the frame in a window named "Cam"
        cv2.imshow("Detecting Face", frame)
        cv2.waitKey(1)

        # Check if 'q' is pressed or more than 10 seconds have passed
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 10:
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

    return face_recognized, match_found

###########################################____DETECT_FACE_ROUTE___################################################

# Updated detect_face route
@app.route('/detect_face/<user_id>', methods=['GET', 'POST'])
def detect_face(user_id):
    global known_face_encodings

    # Open a video capture object for the default camera (index 0)
    video_capture = cv2.VideoCapture(0)

    # Load known faces for the current user
    user_known_face_names, user_known_face_encodings = load_known_faces_for_user(user_id)

    # Call the function to detect and recognize faces
    face_recognized, match_found = detect_and_recognize_face(video_capture, user_id, user_known_face_encodings)

    # Release the video capture object and close windows
    video_capture.release()
    cv2.destroyAllWindows()

    if face_recognized:
        if match_found:
            flash("Your face has been successfully matched.")
            return redirect(url_for('voting_page',user_id=user_id))
        else:
            flash("Face recognized but not matched. Please try again.")
            session['login_error'] = "Face recognized but not matched. Please try again."
            return redirect(url_for('home', user_id=user_id))
    else:
        flash("Face not recognized. Please try again.")
        session['login_error'] = "Face not recognized. Please try again."
        return redirect(url_for('home', user_id=user_id))

##############################################____VOTING_PAGE____############################################

# Store the last vote timestamp for each user in-memory
last_vote_timestamps = {}

# Your existing vote route
@app.route('/voting_page/<user_id>', methods=['GET', 'POST'])
def voting_page(user_id):
    remaining_time = 0  # Initialize remaining_time with a default value

    # Check if the user has voted recently
    if user_id in last_vote_timestamps:
        last_vote_time = last_vote_timestamps.get(user_id, None)
        if last_vote_time:
            cooldown_duration = timedelta(seconds=60)
            elapsed_time = datetime.now() - last_vote_time

            if elapsed_time < cooldown_duration:
                remaining_time = cooldown_duration - elapsed_time
                return redirect(url_for('confirmation_receipt', user_id=user_id))

    if request.method == 'POST':
        # Retrieve the candidate ID from the request JSON
        data = request.get_json()
        candidate_id = data.get('candidate_id')

        # Update the user's voted_party field within the existing user node
        user_ref = db.reference(f'users/{user_id}')
        user_data = user_ref.get()

        if user_data and 'voted_party' in user_data:
            # Check if the user is changing their vote
            if user_data['voted_party'] == candidate_id:
                #flash("You have already voted for this candidate.", 'error')
                return jsonify({"message": "You have already voted for this candidate."}), 400

        user_ref.update({'voted_party': candidate_id})
        last_vote_timestamps[user_id] = datetime.now()

        # Set a session variable to indicate that the user has voted
        session['voted'] = True

        #flash("Vote successfully casted!", 'success')
        return jsonify({"success": True, "message": "Vote successfully casted!"})

    # Check if the user has already voted
    user_ref = db.reference(f'users/{user_id}')
    user_data = user_ref.get()

    if user_data and 'voted_party' in user_data:
        # User has already voted, check the time difference
        last_vote_time = last_vote_timestamps.get(user_id, None)
        if last_vote_time:
            cooldown_duration = timedelta(seconds=60)
            elapsed_time = datetime.now() - last_vote_time

            if elapsed_time < cooldown_duration:
                # User has voted recently, set remaining_time
                remaining_time = cooldown_duration - elapsed_time
                #flash(f"You have already voted", 'error')

    # Render the template with user_id and remaining_time
    return render_template('voting_page.html', user_id=user_id, remaining_time=remaining_time)

##########################################____CONFIRMATION_RECEIPT____########################################

# Specify the path to the Firebase Realtime Database
database_path = '/users'
root_ref = db.reference(database_path)

@app.route('/confirmation_receipt/<user_id>', methods=['GET','POST'])
def confirmation_receipt(user_id):
    user_ref = db.reference(f'users/{user_id}')
    user_data = user_ref.get()

    if user_data and 'voted_party' in user_data:
        # Get user details
        user_name = user_data.get('name', 'Unknown')
        user_age = user_data.get('age', 'Unknown')
        voter_id = user_data.get('voter_id','Unknown')
        # Get voted party
        voted_party = user_data['voted_party']

        # Construct the user's image URL based on voter_id (similar to home route)
        user_image_url = f"https://firebasestorage.googleapis.com/v0/b/{config['storageBucket']}/o/{voter_id}.jpg?alt=media"

        return render_template('confirmation_receipt.html', user_name=user_name, user_age=user_age, voter_id=voter_id, voted_party=voted_party, user_image_url=user_image_url, message=None)
    else:
        message = "You have not voted yet."
        return render_template('confirmation_receipt.html',voter_id=user_id, message=message)

###########################################__EXCEL_FILE__#############################################

def upload_to_storage(data_frame, storage_path):
    try:
        # Convert DataFrame to CSV string
        csv_string = data_frame.to_csv(index=False)

        # Get a reference to the storage bucket
        bucket = storage.bucket()

        # Specify the path to the blob
        blob = bucket.blob(storage_path)

        # Upload CSV string to storage
        blob.upload_from_string(csv_string, content_type='application/csv')

        return blob.public_url
    except Exception as e:
        print(f"Error uploading to Firebase Storage: {e}")
        return None

def create_and_update_csv():
    try:
        # Fetch user data from Firebase
        users_ref = db.reference('/users')
        user_data = users_ref.get()

        if user_data:
            # Create a list to store user information
            user_list = []

            # Loop through each user and extract information
            for voter_id, user_info in user_data.items():
                user_info['voter_id'] = voter_id  # Add voter_id to user_info
                user_list.append(user_info)

            # Convert the list to a DataFrame
            df = pd.DataFrame(user_list)

            # Upload DataFrame to Firebase Storage
            storage_url = upload_to_storage(df[['voter_id', 'name', 'age', 'voted_party']], 'CSV_file/all_users_data.csv')

            if storage_url:
                print("Data exported to CSV and stored in Firebase Storage successfully.")
            else:
                print("Failed to upload CSV file to Firebase Storage.")

        else:
            print("No user data found in the Realtime Database.")

    except Exception as e:
        print(f"Error creating/updating CSV file: {e}")

def on_database_change(event):
    # This function will be triggered whenever there is a change in the Realtime Database
    create_and_update_csv()

# Set up the listener to trigger on any change in the database
root_ref.listen(on_database_change)

# Background function to continuously check for updates and update CSV
def update_csv_background():
    # The background thread will now wait for changes in the Realtime Database and update the CSV accordingly
    while True:
        time.sleep(60)

# Start the background thread
update_thread = Thread(target=update_csv_background)
update_thread.start()

######################################___ABOUT.HTML___#####################################################

@app.route('/about')
def about():
    # Logic for the about page
    return render_template('about.html')

#####################################___LOGOUT_ROUTE___#####################################################

@app.route('/logout')
def logout():
    # Clear the session data (logout the user)
    session.clear()
    # Redirect the user to the login page (replace 'login' with your actual login route)
    return redirect(url_for('login'))
#________________________________________________________________________________________________________

if __name__ == '__main__':
    app.run(debug=True)
