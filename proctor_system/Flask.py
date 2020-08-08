# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/


# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask
import numpy as np
import cv2
from tkinter import *
from tkinter import messagebox
import face_recognition
from flask import Flask, jsonify, request, redirect, render_template, Response
import cv2
import win32api
import os
import numpy as np
# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('p1.html')


@app.route('/work')
def work():
    return render_template('work.html')

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method == 'POST':
        return redirect(url_for('check'))
    return render_template('login.html')

@app.route('/check',methods=['POST','GET'])
def check():
        return detect()

@app.route('/run',methods=['POST','GET'])
def run():
        return runn()


Name="potti"
is_name = False
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

KNOWN_FACES_DIR = "examples/knn_examples/train"

TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog" #use cnn if on GPU or else hog



# Get a reference to webcam #0 (the default one)
cap = cv2.VideoCapture(0)
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR,name)):
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR,name,filename))
        #print(os.path.join(KNOWN_FACES_DIR,name,filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

 
f=0

def detect():
    ind = known_names.index(Name)
    print(ind)
    #print(known_names.index[ind])
    encod = known_faces[ind]
    known_face_encoding = encod
    face_found = False
    is_name = False
    c = 0
    while True:
        ret, image = cap.read()
        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)
        c = c+1
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        f = 0
        for face_encoding, face_location in zip(encodings,locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            match = None

            if True in results:
                face_found = True
                match = known_names[results.index(True)]
                print("Match found : ", match)
                

                if match == Name:
                    is_name = True
                    f=1
                    break
                else:
                    if c==10:
                        is_name = False
                        break
                top_left = (face_location[3],face_location[0])
                bottom_right = (face_location[1], face_location[2])

                color = [0,255,0]

                top_left =  (face_location[3], face_location[2])
                bottom_right = (face_location[1],  face_location[2]+22)

                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match,(face_location[3]+10, face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200))
                
        cv2.imshow(filename, image)
        if f == 1:
            return render_template('work.html')

        else:
            if c==10:
                break
        
            
        # Hit 'q' on the keyboard to quit!
   #    if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()

def runn():
    print("entered")
    root = Tk()

    confidenceThreshold = 0.5
    NMSThreshold = 0.3

    modelConfiguration = 'cfg/yolov3.cfg'
    modelWeights = 'yolov3.weights'

    labelsPath = 'dataset.names'
    labels = open(labelsPath).read().strip().split('\n')

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    outputLayer = net.getLayerNames()
    outputLayer = [outputLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    video_capture = cv2.VideoCapture(0)

    (W, H) = (None, None)

    while True:
        phone = 0
        p = 0
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        if W is None or H is None:
            (H,W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
        net.setInput(blob)
        layersOutputs = net.forward(outputLayer)

        boxes = []
        confidences = []
        classIDs = []

        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    #Apply Non Maxima Suppression
        detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
        if(len(detectionNMS) > 0):
            for i in detectionNMS.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                if labels[classIDs[i]]=="cell phone":
                    messagebox.showwarning("Warning","Phone Detected!!!!")

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    phone = phone + 1
                    text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if labels[classIDs[i]] == "person":
                    p = p + 1
                    if p >=2:
                        win32api.MessageBox(0, 'warning', 'title', 0x00001000)
                        messagebox.showwarning("Warning","Multiple persons detected !!!!")
                    
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, "Persons count"+str(p), (x,y+100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #cv2.imshow('Output', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

#Finally when video capture is over, release the video capture and destroyAllWindows
    video_capture.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
