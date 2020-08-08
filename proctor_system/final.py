from flask import Flask, render_template, request, redirect,url_for
from flask_mysqldb import MySQL
import win32api
import numpy as np
import cv2
import face_recognition
import cv2
import os

app = Flask(__name__)

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

# Configure db
app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "test"

mysql = MySQL(app)

def convertToBinaryData(filename):
    with open(filename,'rb') as file:
        binarydata=file.read()
    return binarydata
 
@app.route('/')
def home(): 
    return render_template('Home.html')

@app.route('/thankyou')
def thankyou(): 
    return render_template('Thank_you.html')
    
@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # Fetch form data
        
        Name = request.form['name']
        mail = request.form['email']
        passwrd= request.form['pass']
        image=request.form['file']
        pic= convertToBinaryData(image)
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO quiz_time(Name, Email,Password,Image) VALUES(%s,%s,%s,%s)",(Name, mail,passwrd,pic))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/guide')
def guide():
    return render_template('Guidelines.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        # Fetch form data
        
        login.Name1= request.form['name']
        mail = request.form['email']
        p= request.form['pass']
        cur = mysql.connection.cursor()
        query = "select * from quiz_time where Email = %s"
        
        result =  cur.execute(query, (mail,))
        records = cur.fetchall()

        if result == 0:
            print("no user")
            win32api.MessageBox(0, 'user not registered', 'warning', 0x00001000)
            
            
        else:
            for row in records:
                password= row[2]
                name = row[0]
          
            
            if password == p:
                if name == login.Name1:
                    print("entered")
                    return redirect(url_for('check'))
                 
                else:
                    win32api.MessageBox(0, 'Invalid Credentials', 'Warning', 0x00001000)                    
                    
            else:
                win32api.MessageBox(0, 'Invalid Credentials', 'Warning', 0x00001000)
            
        
    return render_template('login.html')

@app.route('/check',methods=['POST','GET'])
def check():
    print("entered")
    return detect()

def detect():

    #face_found = False
    #is_name = False
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
                

                if match == login.Name1:
                    #is_name = True
                    f=1
                    break
                else:
                    if c==10:
                        #is_name = False
                        break
                top_left = (face_location[3],face_location[0])
                bottom_right = (face_location[1], face_location[2])

                color = [0,255,0]

                top_left =  (face_location[3], face_location[2])
                bottom_right = (face_location[1],  face_location[2]+22)

                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match,(face_location[3]+10, face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200))
                
        #cv2.imshow(filename, image
        if f == 1:
            print("logged in")
            win32api.MessageBox(0, 'Logged In', 'Success', 0x00001000)
            cap.release()
            cv2.destroyAllWindows()
            return redirect(url_for('guide'))

        else:
            if c==10:
                win32api.MessageBox(0, 'Invalid Examinee', 'Warning', 0x00001000)
                cap.release()
                cv2.destroyAllWindows()
                print("Invalid Examinee")
                return redirect(url_for('home'))
        
            
        # Hit 'q' on the keyboard to quit!
   #    if cv2.waitKey(1) & 0xFF == ord('q'):
            #break


    

#@app.route('/quiz',methods=['POST','GET'])
#def quiz(): 
#   return render_template('quiz.html')

@app.route('/run',methods=['POST','GET'])
def run():
    if request.method == 'POST':
        return redirect(url_for('checkk'))
    return render_template('quiz.html')

@app.route('/checkk')
def checkk():
    return runn()

def runn():
    print("hereeeeeeeee")

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
        person = 0
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
                    phone=phone+1
                    if phone==3:
                        win32api.MessageBox(0, 'If phone is detected once again,you will be logged out of exam', 'Warning', 0x00001000)
                    if phone>3:
                        return redirect(url_for('home'))
                        
                    win32api.MessageBox(0, 'Phone Detected!Put Away', 'Warning', 0x00001000)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    phone = phone + 1
                    text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if labels[classIDs[i]] == "person":
                    p = p + 1
                    pic=frame[y:y+h,x:x+w]
                    cv2.imwrite('pic_1.png',pic)  # unkknown person image 
                    if p >=2:
                        person=person+1
                        if phone==3:
                            win32api.MessageBox(0, 'If two people are detected once again,you will be logged out of exam', 'Warning', 0x00001000)
                        if phone>3:
                            return redirect(url_for('home'))
                        win32api.MessageBox(0, 'Multiple Persons Detected', 'Warning', 0x00001000)
                    elif p==1:
                        # Load the jpg files into numpy arrays
                        image_file = login.Name1+"/"+str(p)+".jpg"
                        
                        known_image = face_recognition.load_image_file(os.path.join("examples/knn_examples/train",image_file))
                        #obama_image = face_recognition.load_image_file("obama.jpg")
                        unknown_image = face_recognition.load_image_file("pic_1.png")

                        # Get the face encodings for each face in each image file
                        # Since there could be more than one face in each image, it returns a list of encodings.
                        # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
                        try:
                            known_face_encoding = face_recognition.face_encodings(known_image)[0]
                            #obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
                            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
                        except IndexError:
                            print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
                            quit()

                        known_faces = [
                            known_face_encoding
                            #obama_face_encoding
                        ]

                        # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
                        results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
                        if results[0]==False:
                            win32api.MessageBox(0, 'You are not the authorized examinee,thereforth you are being logged out of exam', 'Warning', 0x00001000)
                            return redirect(url_for('home'))
                            
                        print("Is the unknown face a picture of {}? {}".format(login.Name1,results[0]))
                        #print("Is the unknown face a picture of Obama? {}".format(results[1]))
                        #print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
     
            
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, "Persons count"+str(p), (x,y+100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #cv2.imshow('Output', frame)
      #  if(cv2.waitKey(1) & 0xFF == ord('q')):
        #    break

#Finally when video capture is over, release the video capture and destroyAllWindows
  #  video_capture.release()
   # cv2.destroyAllWindows()

@app.route('/finish')
def finish():
    video_capture.release()
    cv2.destroyAllWindows()
    return render_template('Thank_you.html')

@app.route('/close')
def close():
    closee()

def closee():
    video_capture.release()
    cv2.destroyAllWindows()
    

if __name__=="__main__":
    app.run(debug=True)
    
 
