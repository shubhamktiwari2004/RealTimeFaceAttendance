from flask import Flask, render_template, request, jsonify, Response, session, redirect
import dlib
import numpy as np
import cv2
import os
import face_recognition
from datetime import datetime, timedelta
from firebase_admin import credentials, initialize_app, storage, db
import requests
import tempfile
from mimetypes import guess_type
import csv
from io import StringIO


########################################################################################################################


cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred,{
    'databaseURL' : "https://realtimefaceattendance-db2a8-default-rtdb.firebaseio.com/",
    'storageBucket': "realtimefaceattendance-db2a8.appspot.com"
})


########################################################################################################################


app = Flask(__name__)

# Reference to the Firebase Storage bucket
bucket = storage.bucket()

# Reference to the Firebase Realtime Database
db_ref = db.reference('faces')

#reference for landmark in realtime database
db_landmark=db.reference('FaceLandamrk')



########################################################################################################################



# Generate a signed URL for the shape predictor file
shape_predictor_blob = bucket.blob("landmark/shape_predictor_68_face_landmarks.dat")

# Set expiration time for the signed URL (e.g., 365 days)
expiration_time = datetime.utcnow() + timedelta(days=365)

# Generate a signed URL
shape_predictor_url = shape_predictor_blob.generate_signed_url(expiration=int(expiration_time.timestamp()))

# Store the generated URL in the Realtime Database
db_landmark.child('dlibmark').update({'landmark_url': shape_predictor_url})
landmark_url = db_landmark.child('dlibmark').child('landmark_url').get()

# Download the shape predictor file using the URL
response = requests.get(shape_predictor_url)
shape_predictor_content = response.content

# Save the shape predictor file to a temporary file
temp_landmark_path = tempfile.NamedTemporaryFile(delete=False)
temp_landmark_path.write(shape_predictor_content)
temp_landmark_path.close()


########################################################################################################################

IMAGE_FOLDER="Images"

app.secret_key = 'secret'


# Login route
@app.route('/login1', methods=['GET', 'POST'])
def login1():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Extract the part of email before @gmail.com
        email_part = email.split('@')[0]

        # Retrieve user data from Realtime Database
        user_ref = db.reference(f'users/{email_part}').get()
        if user_ref and user_ref.get('password') == password:
            session['user'] = email  # Set user ID in session
            return redirect('/FR_templates')
        else:
            message='Invalid credentials'
            return render_template('login.html', message=message)
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Remove the 'user' key from the session if it exists
    session.pop('user', None)
    # Redirect the user to the login page
    return redirect('/')

# Define a route to render the HTML page
@app.route('/', methods=['GET','POST'])
def login():
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    return render_template('register.html')

@app.route('/FR_templates')
def FR_templates():
    if 'user' in session:
        current_user_email = session['user']
        return render_template('/FR_templates.html', current_user_email=current_user_email)
    else:
        return redirect('/login1')

@app.route('/camera')
def camera():
    if 'user' in session:
        current_user_email = session['user']
        return render_template('/camera.html',current_user_email=current_user_email)
    else:
        return redirect('/login1')

@app.route('/about')
def about():
    if 'user' in session:
        current_user_email = session['user']
        return render_template('/about.html', current_user_email=current_user_email)
    else:
        return redirect('/login1')
########################################################################################################################

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image = request.files['image']

        # Specify the folder path in Firebase Storage with a predefined extension
        blob_path = f'{IMAGE_FOLDER}/{image.filename}.jpg'
        blob = bucket.blob(blob_path)

        # Determine the MIME type of the image
        content_type, _ = guess_type(image.filename)
        if content_type is not None:
            blob.upload_from_file(image, content_type=content_type)
        else:
            # Default to 'image/jpeg' if MIME type cannot be determined
            blob.upload_from_file(image, content_type='image/jpeg')

        return jsonify({"success": True, "message": "Image uploaded successfully"})

    return jsonify({"success": False, "message": "Image not found in the request"})
########################################################################################################################

@app.route('/download_csv')
def download_csv():
    # Fetch data from the 'attendance' node in the Realtime Database
    attendance_data = db_attendance.get()

    # Prepare CSV content
    csv_content = StringIO()
    csv_writer = csv.writer(csv_content)

    # Write header
    csv_writer.writerow(['Name', 'Date', 'Time', 'Attendance'])

    # Write data rows
    for name, data in attendance_data.items():
        csv_writer.writerow([data['name'], data['date'], data['time'], data['attendance']])

    # Prepare response
    response = Response(csv_content.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=attendance_data.csv'

    return response
########################################################################################################################

db_attendance=db.reference('attendance')

def mark_attendance(name, status):
    current_datetime = datetime.now()

    # Check if the name exists in the attendance records
    student_info = db_attendance.child(name).get()

    if student_info:
        last_attendance_date = student_info.get('date', '1970-01-01')
        last_attendance_time = student_info.get('time', '00:00:00')

        # Convert last_attendance_date and last_attendance_time to datetime objects
        last_datetime = datetime.strptime(f"{last_attendance_date} {last_attendance_time}", "%Y-%m-%d %H:%M:%S")

        # Calculate the time difference in seconds
        time_diff = (current_datetime - last_datetime).total_seconds()

        if time_diff > 30 or current_datetime.date() > last_datetime.date():
            # Update the last attendance date and time if 30 seconds have passed or it's a new day
            db_attendance.child(name).child('date').set(current_datetime.strftime("%Y-%m-%d"))
            db_attendance.child(name).child('time').set(current_datetime.strftime("%H:%M:%S"))
    else:
        # If the name does not exist, add the attendance record
        attendance_ref = db_attendance.child(name)
        attendance_ref.set({
            'name': name,
            'date': current_datetime.strftime("%Y-%m-%d"),
            'time': current_datetime.strftime("%H:%M:%S"),
            'attendance': status  # (Present or Absent)
        })
########################################################################################################################


def perform_facial_recognition():
    # Function to generate image URLs for all images in the 'Images' folder
    def get_image_urls_from_storage():
        image_urls = {}
        blobs = bucket.list_blobs(prefix='Images/')

        for blob in blobs:
            # Generate a signed URL with a very distant future expiration
            expiration_time =datetime.utcnow()+timedelta(days=365)
            # Generate a signed URL that can be publicly accessed
            image_url = blob.generate_signed_url(expiration=int(expiration_time.timestamp()))

            # Extract the image filename from the full path
            _, image_filename = os.path.split(blob.name)

            # Store the image URL in the dictionary using the filename as the key
            image_urls[image_filename] = image_url

        return image_urls

    # Use the storage database directly
    image_urls = get_image_urls_from_storage()

    # Initialize an empty list for images
    images = []
    className = []

    # Use image URLs directly instead of reading from the local file system
    for cls, image_url in image_urls.items():
        # You might need to adjust this part based on your requirements.
        response = requests.get(image_url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        images.append(img)
        className.append(os.path.splitext(cls)[0])

        # Update the Realtime Database with the image URL
        db_ref.child(className[-1]).update({'image_url': image_url})

    print(className)




    def Normalize(img):
        if len(img.shape) > 2 and img.shape[2] > 1:
            normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img

    def findEncodings(images):
        encodings_img = []
        for img in images:
            img = Normalize(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodings_img.append(encode)
        return encodings_img

    encodingImgList = findEncodings(images)


    # print(encodingImgList)

    # Loop through the available camera indexes and check for frames
    # for i in range(10):
    #     cap = cv2.VideoCapture(i)
    #     if cap.isOpened():
    #        print(f"Camera index: {i} - Opened")
    #         cap.release()
    #     else:
    #         print(f"Camera index: {i} - Closed")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.abspath(temp_landmark_path.name))
    cap = cv2.VideoCapture(0)
    frames_to_process = 60

    for _ in range(frames_to_process):
        success, img = cap.read()
        if not success:
            break


    recognized_names = []
    while True:
        success, img = cap.read()
        img = Normalize(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(img)
        current_face = face_recognition.face_locations(img)
        curr_face_encoding = face_recognition.face_encodings(img, current_face)

        for face in faces:
            landmarks = predictor(img, face)
            # Draw the facial landmarks on the frame
            for point in landmarks.parts():
                cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), -1)




        for encodeFace, faceLoc in zip(curr_face_encoding, current_face):
            matches = face_recognition.compare_faces(encodingImgList, encodeFace)
            faceDistance = face_recognition.face_distance(encodingImgList, encodeFace)
            # print(faceDistance)
            match = np.argmin(faceDistance)

            if matches[match]:
                name = className[match].upper()
                recognized_names.append(name)

                # print(name)
                max = 1
                accuracy = (1 - faceDistance[match]) * 100
                Accuracy = round(accuracy, 2)
                y1, x2, y2, x1 = faceLoc
                if Accuracy > 48:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
                    cv2.putText(img, f"{name} {str(Accuracy)}%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    # cv2.putText(img, str(Accuracy), (x1 - 200, y1 - 115), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
                    # Calling the function
                    mark_attendance(name, "Present")

        # for name in className:
        #     name=name.upper()
        #     if name not in recognized_names:
        #         # Mark attendance as absent for the unrecognized name
        #         mark_attendance(name, "Absent")




        ret, jpeg_buffer = cv2.imencode('.jpg', img)
        jpeg_frame = jpeg_buffer.tobytes()
        ret, png_buffer = cv2.imencode('.png', img)
        png_frame = png_buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')

        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + png_frame + b'\r\n')

########################################################################################################################

@app.route('/video_feed')
def video_feed():
    return Response(perform_facial_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)


