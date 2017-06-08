#!/usr/bin/env python

import sys
import os
import numpy as np
import cv2

cascPath = "face_recognition_system/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture =  cv2.VideoCapture(0)


def resize(images, size=(100, 100)):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # using different OpenCV method if enlarging or shrinking
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    
    return images_norm

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm


def cut_face_rectangle(image, face_coord):
    images_rectangle = []
    for (x, y, w, h) in face_coord:
        images_rectangle.append(image[y: y + h, x: x + w])
    return images_rectangle

def draw_face_rectangle(image, faces_coord):
    for (x, y, w, h) in faces_coord:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def blur_faces(img,faces_coord):
    for (x, y, w, h) in faces_coord:
        cropped = img[y:y+h, x:x+w]
        cropped = cv2.blur(cropped, (23, 23))
        img[y:y+h, x:x+w] = cropped

def detect(image, biggest_only=True):
    #self.classifier = cv2.CascadeClassifier(xml_path)
    # is_color = len(image) == 3
    #   if is_color:
    #       image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #   else:
        image_gray = image
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        flag = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                                                    cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                                                    cv2.CASCADE_SCALE_IMAGE
        face_coord = faceCascade.detectMultiScale(
                                                        image_gray,
                                                        scaleFactor=scale_factor,
                                                        minNeighbors=min_neighbors,
                                                        minSize=min_size,
                                                        flags=flag
                                                    )
                                                            
        return face_coord


def get_images(frame, faces_coord):
    faces_img = cut_face_rectangle(frame, faces_coord)
    frame = draw_face_rectangle(frame, faces_coord)
    faces_img = normalize_intensity(faces_img)
    faces_img = resize(faces_img)
    return (frame, faces_img)

def add_person(people_folder):
    person_name = input('What is the name of the new person: ').lower()
    folder = people_folder + person_name
    if not os.path.exists(folder):
        input("I will now take 20 pictures. Press ENTER when ready.")
        os.mkdir(folder)
        #video = cv2.VideoCapture(0)
        #detector = FaceDetector('face_recognition_system/frontal_face.xml')
        faceCascade = cv2.CascadeClassifier(cascPath)
        counter = 1
        timer = 0
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
        while counter < 21:
            ret,frame = video_capture.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            face_coord = detect(gray_frame,True)
            if len(face_coord):
                gray_frame, face_img = get_images(gray_frame, face_coord)
                
                if timer % 100 == 10:
                    cv2.imwrite(folder + '/' + str(counter) + '.jpg',
                                face_img[0])
                    print ('Images Saved:', str(counter))
                    counter += 1
                    cv2.imshow('Saved Face', face_img[0])

            cv2.imshow('Video Feed', gray_frame)
            cv2.waitKey(50)
            timer += 10
            if cv2.waitKey(100) & 0xFF == 27:
                sys.exit()


    else:
        print ("This name already exists!")
        sys.exit()
    if cv2.waitKey(100) & 0xFF == 27:
        sys.exit()


def recognize_people(people_folder):
    """ Start recognizing people in a live stream with your webcam
    """
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print ("Have you added at least one person to the system?")
        sys.exit()
    print ("These are the people in the Recognition System:")
    for person in people:
        print ("-" , person)

    recognizer = cv2.face.createLBPHFaceRecognizer()
    threshold = 95
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            print ("image: " , labels)
            labels.append(i)
    try:
        recognizer.train(images, np.array(labels))
    except:
        print ("\nOpenCV Error: Do you have at least two people in the folder?\n")
        sys.exit()

    video = cv2.VideoCapture(0)

    while True:
        ret,frame = video_capture.read()
        faces_coord = detect(frame, False)
        if len(faces_coord):
            frame, faces_img = get_images(frame, faces_coord)
            for i, face_img in enumerate(faces_img):
                pred, conf = recognizer.predict(face_img)
                print ("Prediction: " , str(pred))
                print("labels:", labels_people[pred])
                print ('Confidence: ' , str(round(conf)))
                print ('Threshold: ' , str(threshold))
                if conf < threshold:
                    cv2.putText(frame, labels_people[pred].capitalize(),
                                (faces_coord[i][0], faces_coord[i][1] - 2),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (0, 255, 0), 2,
                                cv2.LINE_AA)
                #  blur_faces(frame, faces_coord)
            
                else:
                    cv2.putText(frame, "Unknown",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (0, 255, 0), 2,
                                cv2.LINE_AA)

        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == 27:
            sys.exit()

def check_choice():
    """ Check if choice is good
    """
    is_valid = 0
    while not is_valid:
        try:
            choice = int(input('Enter your choice [1-3] : '))
            if choice in [1, 2, 3]:
                is_valid = 1
            else:
                print ("'%d' is not an option.\n" % choice)
        except ValueError:
            print ("Invaid value!")
    return choice

if __name__ == '__main__':
    print (30 * '-')
    print ("   Pick One")
    print (30 * '-')
    print ("1. Add person to the recognizer system")
    print ("2. Start recognizer")
    print ("3. Exit")
    print (30 * '-')

    CHOICE = check_choice()

    PEOPLE_FOLDER = "face_recognition_system/people/"

    if CHOICE == 1:
        if not os.path.exists(PEOPLE_FOLDER):
            os.makedirs(PEOPLE_FOLDER)
        add_person(PEOPLE_FOLDER)
    elif CHOICE == 2:
        recognize_people(PEOPLE_FOLDER)
    elif CHOICE == 3:
        sys.exit()
