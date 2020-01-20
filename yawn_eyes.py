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

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(36,39):
        top_lip_pts.append(landmarks[i])
    # for i in range(61,64):
    #     top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(40,41):
        bottom_lip_pts.append(landmarks[i])
    # for i in range(56,59):
    #     bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        print("Person is not facing the camera peroperly")
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    

cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False 

while True:
    ret, frame = cap.read()   
    image_landmarks, lip_distance = mouth_open(frame)
    
    prev_yawn_status = yawn_status  
    
    if lip_distance < 5: #detecting the lips distance longers than 10px
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
    cv2.imshow('Yawn Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 means Press Enter to Exit
        break
        
cap.release()
cv2.destroyAllWindows() 
