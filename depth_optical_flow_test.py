import cv2 as cv
import time
import math
from core_modules.optical_flow import detect_motion
from core_modules.stereo_depth_sensing import generate_depth_frame

#TAKING CAMERA FEED
cap1 = cv.VideoCapture(2)  # Change according to camera input
cap2 = cv.VideoCapture(4) # Change according to camera input

#initializing parameters

h_fov= 70#in degrees
v_fov= 45.83# in degrees
#converting fovs to radians
h_fov=h_fov*(math.pi/180)
v_fov=v_fov*(math.pi/180)
altitude= 3 # in metres


if not (cap1.isOpened() or cap2.isOpened()) :
    print("Error: Could not open camera.")
    exit()

ret1, l_frame = cap1.read()
ret2, r_frame = cap2.read()
e_time1=time.time()
if not (ret1 or ret2):
    print("Error: Failed to capture initial frame.")
    cap1.release()
    cap2.release()
    exit()

depth_frame1=generate_depth_frame(l_frame, r_frame)

#RUNNING FLOW ANALYSIS ON DEPTH FRAME
while True:
    ret1, l_frame = cap1.read()
    ret2, r_frame = cap2.read()
    if not (ret2 or ret1):
        print("Error: Failed to capture frame.")
        break
    

    #CORE PROGRAM
    depth_frame2=generate_depth_frame(l_frame, r_frame)
    e_time2=time.time()
    velocity_vector, angular_velocity=detect_motion(depth_frame1, depth_frame2,h_fov, v_fov,altitude, (1/(e_time2-e_time1)))
    
    cv.imshow("Depth Frame", depth_frame2)
    print("X velocity: ",velocity_vector[0]*math.cos(velocity_vector[1]),"\t Y velocity: ", velocity_vector[0]*math.sin(velocity_vector[1])*-1, "\t Angular Velocity: ", angular_velocity, " CCW"if(angular_velocity>0)else " CW")


    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
    
    #updating prev_frame and time
    depth_frame1 = depth_frame2.copy()
    e_time1=e_time2

cap1.release()
cap2.release()
cv.destroyAllWindows()

