import face_recognition
import cv2
import numpy as np
import os
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
KNOWN_FACES_DIR = "examples/knn_examples/train"

TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" #use cnn if on GPU or else hog



# Get a reference to webcam #0 (the default one)
cap = cv2.VideoCapture(0)

print(" loading known faces...................")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR,name)):
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR,name,filename))
        print(os.path.join(KNOWN_FACES_DIR,name,filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
        


print("processing unknwn faces")


while True:
    # Grab a single frame of video
    ret, image = cap.read()

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    

    for face_encoding, face_location in zip(encodings,locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print("Match found : ", match)
            top_left = (face_location[3],face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0,255,0]

            top_left =  (face_location[3], face_location[2])
            bottom_right = (face_location[1],  face_location[2]+22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match,(face_location[3]+10, face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200))

                        
            
      
    cv2.imshow(filename, image)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
