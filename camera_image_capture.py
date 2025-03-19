import cv2

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
i=0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 'q' to quit
        cv2.imwrite("img"+str(i)+".png",cv2.rotate(frame, cv2.ROTATE_180))
        i+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
