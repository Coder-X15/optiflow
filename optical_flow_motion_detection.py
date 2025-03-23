import cv2
import math

def compute_net_flow(flow_magnitude, flow_direction):
    x_net = 0
    y_net = 0
    for i in range(flow_magnitude.shape[0]):
        for j in range(flow_magnitude.shape[1]):
            x_net += flow_magnitude[i, j] * math.cos(flow_direction[i, j])
            y_net += flow_magnitude[i, j] * math.sin(flow_direction[i, j]) * -1
    return (math.sqrt(x_net**2 + y_net**2), math.atan2(y_net, x_net))

def display_net_flow(net_grid):
    for i in range(len(net_grid)):
        for j in range(len(net_grid[i])):
            net_angle = net_grid[i][j][-1]

            if (0 <= net_angle < (math.pi)/4) or ((7*math.pi/4) <= net_angle < (2*math.pi)):
                print("right", end=" ")
            elif (math.pi/4 <= net_angle < 3*(math.pi)/4):
                print("up", end=" ")
            elif (3*math.pi/4 <= net_angle < (5*math.pi)/4):
                print("left", end=" ")
            else:
                print("down", end=" ")
        print()

def detect_motion(frame1, frame2):
    # Resizing frames
    scale = 0.5
    frame1_resized = cv2.resize(frame1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    frame2_resized = cv2.resize(frame2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # Detecting flow using PCA
    pca_flow = cv2.optflow.createOptFlow_PCAFlow()
    flow = pca_flow.calc(gray1, gray2, None)
    flow_magnitude, flow_angles = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Removing noise
    threshold = 1.2
    flow_magnitude[flow_magnitude < threshold] = 0

    height, width = flow_magnitude.shape
    y_step = height // 3
    x_step = width // 3

    # Generating grids
    grids_direction = [[None]*3 for _ in range(3)]
    grids_mag = [[None]*3 for _ in range(3)]

    for i in range(3):
        for j in range(3):
            grids_direction[i][j] = flow_angles[i*y_step:(i+1)*y_step, j*x_step:(j+1)*x_step]
            grids_mag[i][j] = flow_magnitude[i*y_step:(i+1)*y_step, j*x_step:(j+1)*x_step]

    net_grid = [[compute_net_flow(grids_mag[i][j], grids_direction[i][j]) for j in range(3)] for i in range(3)]
    
    print()
    display_net_flow(net_grid)
    print()
    
    return net_grid

# MAIN CODE
cap = cv2.VideoCapture(2)  # Change according to camera input
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret1, frame1 = cap.read()
if not ret1:
    print("Error: Failed to capture initial frame.")
    cap.release()
    exit()

while True:
    ret2, frame2 = cap.read()
    
    if not ret2:
        print("Error: Failed to capture frame.")
        break

    # Detecting flow
    motion_grid = detect_motion(frame1, frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    frame1 = frame2.copy()

cap.release()
cv2.destroyAllWindows()
