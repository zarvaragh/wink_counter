import cv2
import dlib
import numpy as np
import sys

path = sys.executable
path = path.replace("pythonw.exe", "shape_predictor_68_face_landmarks.dat")

# path = 'C:\\Users\\RAVEN\\Anaconda3\\envs\\tensorflow1_cpu\\shape_predictor_68_face_landmarks.dat'

predictor = dlib.shape_predictor(path)
detector = dlib.get_frontal_face_detector()


def get_landmarks(img):
    rects = detector(img, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img

def top_eyelash(landmarks):
    top_eyelash_pts = []
    for i in range(36,39):
        top_eyelash_pts.append(landmarks[i])
    # for i in range(61,64):
    #     top_eyelash_pts.append(landmarks[i])
    top_eyelash_mean = np.mean(top_eyelash_pts, axis=0)
    return int(top_eyelash_mean[:,1])

def bottom_eyelash(landmarks):
    bottom_eyelash_pts = []
    for i in range(40,41):
        bottom_eyelash_pts.append(landmarks[i])
    # for i in range(56,59):
    #     bottom_eyelash_pts.append(landmarks[i])
    bottom_eyelash_mean = np.mean(bottom_eyelash_pts, axis=0)
    return int(bottom_eyelash_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        print("Person is not facing the camera peroperly")
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_eyelash_center = top_eyelash(landmarks)
    bottom_eyelash_center = bottom_eyelash(landmarks)
    eyelash_distance = abs(top_eyelash_center - bottom_eyelash_center)
    return image_with_landmarks, eyelash_distance


#capturing the video from webcam 
cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False 

while True:
    ret, frame = cap.read()   
    image_landmarks, eyelash_distance = mouth_open(frame)
    
    prev_yawn_status = yawn_status  
    
    if eyelash_distance < 5: #detecting the eyelashs distance smaller than 5px
        yawn_status = True 
        
        cv2.putText(frame, "Subject is Yawning", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

        output_text = " Yawn Count: " + str(yawns + 1)

        cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Wink Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 means Press Enter to Exit
        break
        
cap.release()
cv2.destroyAllWindows() 
