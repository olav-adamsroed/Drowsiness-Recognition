import face_recognition
import cv2
from face_recognition.api import batch_face_locations
import numpy as np

vid_capture = cv2.VideoCapture(0)

frames = []
frame_count = 0


while vid_capture.isOpened():
    ret, frame = vid_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if not ret:
        break

    small_frame = small_frame[:, :, ::-1]

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)


    frame_count += 1
    frames.append(small_frame)

    if len(frames) == 128:
        batch_of_face_locations = face_recognition.batch_face_locations(small_frames, number_of_times_to_upsample=0)

        # Now let's list all the faces we found in all 128 frames
        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            frame_number = frame_count - 128 + frame_number_in_batch
            print("I found {} face(s) in frame #{}.".format(number_of_faces_in_frame, frame_number))

            for face_location in face_locations:
                # Print the location of each face in this frame
                top, right, bottom, left = face_location
                print(" - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Clear the frames array to start the next batch
        frames = []


""" 

# initialize some variables for face detection
face_locations = []
face_encodings = []
process_this_frame = True

while True:

    # grabs a single frame of video
    ret, frame = vid_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition proccesing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy = 0.25)

    # Convert the image from BGR color ( OpenCV ) to RGB color ( race_recognition ).
    rgb_small_frame = small_frame[:, :, ::-1]


    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)


"""
