import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
from scipy.spatial import distance
from datetime import datetime, time

#calculate eye aspect ratio, how open/closed the eye is
# 6 points in each eye
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

 #mouth = topLip[10], topLip[9], topLip[8], bottomLip[10], bottomLip[9], bottomLip[8]

 # calculate mouth aspect ratio, how open/closed the mouth is
 # inner mouth has 6 points, 3 points above and 3 points below
 # We find the distance between each corresponding lower/upper point
 # add all 3 values together and divide sum by 3
def mouth_aspect_ratio(top, bottom): 
	A = distance.euclidean(top[10], bottom[8])
	B = distance.euclidean(top[9], bottom[9])
	C = distance.euclidean(top[8], bottom[10])

	MAR = (A + B + C) / 3.0
	return MAR


def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"


def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip_method} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

# Should take in yawns
def time_counter(MAR):

    yawn_check = MAR
    yawn_count = []
    yawn_compare = []
    time_count = 0
    time_spent = timer_end - timer_start
    timer_start= time.time()
    print("this should be our if statement where we count some time lol")
    
    if timer_end - time_start == 59:
        yawn_count[i] = yawn_check
        i += 1

        if yawn_check[i] == 9:
            yawn_compare[i]
       

    timer_end = time.time()
    
    time_subtracter = datetime.combine(now.date(), time(0))
    print(timer - time_subtractor)



def main_loop():
    # Get access to the webcam. The method is different depending on if this is running on a laptop or a Jetson Nano.
    if running_on_jetson_nano():
        # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
        video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    else:
        # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
        # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
        video_capture = cv2.VideoCapture(0)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    width  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)   
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("width", width)
    print("height", height)

    process_this_frame = True
    i = 0
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # list with posisitions of points making up the facial features (eyes, mouth, etc)
        face_landmarks_list = face_recognition.face_landmarks(frame)

        # her er koden som finner øyene og markerer de på skjermen
        for face_landmarks in face_landmarks_list:
                
            # array som inneholde punktene som gjør opp øyene
            # lefteye[0-5] og righteye[0-5], alle inneholder (x, y) posisjon
            leftEye = np.array(face_landmarks['left_eye'])
            rightEye = np.array(face_landmarks['right_eye'])

            topLip = np.array(face_landmarks['top_lip'])
            bottomLip = np.array(face_landmarks['bottom_lip'])

            #mouth = np.array([topLip[10], topLip[9], topLip[8], bottomLip[10], bottomLip[9], bottomLip[8]])
                    
            # caller def eye_aspect_ratio på begge øyene
            # denne returner en verdi = EAR (eye aspect ratio) for hvert øye
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            mar = mouth_aspect_ratio(topLip, bottomLip)
            #yawn_counter = time_counter(mar)

            # total EAR ( eye aspect ratio) på begge øyene
            # EAR brukes senere til å se hvor åpne øyene er
            ear = (leftEAR + rightEAR) / 2.0

            marstring = 'MAR: ' + str("{:.2f}".format(mar))
            earstring = 'EAR: ' + str("{:.2f}".format(ear))
            cv2.putText(frame, marstring, (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, earstring, (450, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            topLipHull = cv2.convexHull(topLip)
            bottomLipHull = cv2.convexHull(bottomLip)

            # tegner grønt rundt leppene
            cv2.drawContours(frame, [topLipHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [bottomLipHull], -1, (0, 255, 0), 1)

            # tegner grønt rundt øyene
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            eye_thresh = 0.20
            mouth_thresh = 30
            frame_check = 30
                    
            flag=i

            #width 640.0
            #height 480.0

            if ear < eye_thresh or mar > mouth_thresh:
                i += 1
                print (flag)
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (100, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print ("Are you Drowsy!?")
                else:
                    flag = 0
       
                    
    # Draw a label with a name below the face
    #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        # We need to save our known faces back to disk every so often in case something crashes.
       # if len(face_locations) > 0 and number_of_faces_since_save > 100:
            #save_known_faces()
            #number_of_faces_since_save = 0
       # else:
            #number_of_faces_since_save += 1

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()