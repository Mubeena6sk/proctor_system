import face_recognition as fr
import os

KNOWN_FACES_DIR = "examples/knn_examples/train"
def compare_faces(file1, file2):
    """
    Compare two images and return True / False for matching.
    """
    # Load the jpg files into numpy arrays
    image1 = fr.load_image_file(file1)
    image2 = fr.load_image_file(file2)
    
    # Get the face encodings for each face in each image file
    # Assume there is only 1 face in each image, so get 1st face of an image.
    image1_encoding = fr.face_encodings(image1)[0]
    image2_encoding = fr.face_encodings(image2)[0]
    
    # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = fr.compare_faces([image1_encoding], image2_encoding)    
    return results[0]

#Each face is tuple of (Name,sample image)    
'''known_faces = [('Obama','examples/knn_examples/train/obama/obama.jpg'),
               ('potti','examples/knn_examples/train/potti/IMG_20200202_131450.jpg'),
              ]'''

known_faces = []
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR,name)):
        known_faces.append(os.path.join(KNOWN_FACES_DIR,name,filename))
'''for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR,name)):
        image = fr.load_image_file(os.path.join(KNOWN_FACES_DIR,name,filename))
        print(os.path.join(KNOWN_FACES_DIR,name,filename))
        encoding = fr.face_encodings(image)[0]
        known_faces.append(encoding)
        #known_names.append(name)'''
        
def face_rec(file):
    """
    Return name for a known face, otherwise return 'Uknown'.
    """
    for known_file in known_faces:
        if compare_faces(known_file,file):
            return 'there'
    return 'Unknown' 
    
def find_facial_features(file):
    # Load the jpg file into a numpy array
    image = fr.load_image_file(file)

    # Find all facial features in all the faces in the image
    face_landmarks_list = fr.face_landmarks(image)
    
    # return facial features if there is only 1 face in the image
    if len(face_landmarks_list) != 1:
        return {}
    else:
        return face_landmarks_list[0]
        
def find_face_locations(file):
    # Load the jpg file into a numpy array
    image = fr.load_image_file(file)

    # Find all face locations for the faces in the image
    face_locations = fr.face_locations(image)
    
    # return facial features if there is only 1 face in the image
    if len(face_locations) != 1:
        return []
    else:
        return face_locations[0]        
