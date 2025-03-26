import cv2
import math

def compute_grid_flow(flow_magnitude, flow_direction):
    ''' INPUT: a 2D flow_magnitude frame  where each cell corresponds to the magnitude of flow vector at the corresponding pixel\tand\n
                another 2D flow_direction frame where each cell corresponds to the angle of flow vector  with repect to x axis at the corresponding pixel
        OUTPUT: a tuple of the form (net_magnitude,direction_angle) of the resultant vector of all flow vectors in the frame'''
    x_net = 0
    y_net = 0

    for i in range(flow_magnitude.shape[0]):
        for j in range(flow_magnitude.shape[1]):
            x_net += flow_magnitude[i, j] * math.cos(flow_direction[i, j])
            y_net += flow_magnitude[i, j] * math.sin(flow_direction[i, j]) * -1

    x_net/=(flow_magnitude.shape[0]*flow_magnitude.shape[1])
    y_net/=(flow_magnitude.shape[0]*flow_magnitude.shape[1])
    return (math.sqrt(x_net**2 + y_net**2),(2*math.pi)- math.atan2(y_net, x_net))


def vector_component(vector1, vector2):
    '''utility method to compute component of vector1 in the direction of vector2'''
    v1_mag, v1_angle = vector1
    v2_mag, v2_angle = vector2
    return v1_mag * math.cos(v2_angle - v1_angle)

def compute_net_flow(grids):
    '''
    INPUT: 3 x 3 array of (flow_magnitude, flow_angle) tuples.
    OUTPUT: a tuple of net_translation_flow_vector in the format (magnitude, angle) and net curl vector corresponding to the net_rotation_flow_vector magnitude.
    '''
    net_flow_x = 0
    net_flow_y = 0

    for row in range(len(grids)):
        for col in range(len(grids[row])):
            net_flow_x += grids[row][col][0] * math.cos(grids[row][col][1])
            net_flow_y += grids[row][col][0] * math.sin(grids[row][col][1]) * -1  # Negating y for correct flow
    net_flow_x/=9
    net_flow_y/=9
    # Computing net translational flow vector
    net_translation_vector = (
        math.sqrt(net_flow_x**2 + net_flow_y**2),
        (2 * math.pi) - math.atan2(net_flow_y, net_flow_x)
    )

    # Computing net rotational flow magnitude
    principal_vector = (1, math.pi)
    next_steps = ['l', 'd', 'r', 'u']
    next_step_index = 0
    next_step = 'l'
    curr_pos = [0, 1]
    net_rotational_flow_magnitude = 0

    for _ in range(8):  #stops within the 3x3 grid
        net_rotational_flow_magnitude += vector_component(grids[curr_pos[0]][curr_pos[1]], principal_vector)

        # Updating next_step
        if next_step == 'l' and curr_pos[1] == 0:
            next_step_index += 1
        elif next_step == 'd' and curr_pos[0] == 2:
            next_step_index += 1
        elif next_step == 'r' and curr_pos[1] == 2:
            next_step_index += 1

        if next_step_index >= len(next_steps):
            break  # Prevents out-of-bounds error

        next_step = next_steps[next_step_index]

        # Updating curr_pos 
        if next_step == 'l' and curr_pos[1] > 0:
            curr_pos[1] -= 1
        elif next_step == 'd' and curr_pos[0] < 2:
            curr_pos[0] += 1
        elif next_step == 'r' and curr_pos[1] < 2:
            curr_pos[1] += 1
        elif next_step == 'u' and curr_pos[0] > 0:
            curr_pos[0] -= 1

        # Updating principal_vector (rotating by +pi/4 ccw)
        principal_vector = (principal_vector[0], principal_vector[1] + (math.pi / 4))

    return net_translation_vector, (net_rotational_flow_magnitude/8)



def display_flow(translational_flow_vector_polar,rotational_flow_magnitude ):
    '''display method for debugging'''

    x_translational_flow_component=translational_flow_vector_polar[0]*math.cos(translational_flow_vector_polar[1])
    y_translational_flow_component=translational_flow_vector_polar[0]*math.sin(translational_flow_vector_polar[1])*-1

    # print("X: ", x_translational_flow_component, "\tY: ", y_translational_flow_component, "\tRotational: ", rotational_flow_magnitude,"(CCW)" if(rotational_flow_magnitude>=0) else "(CW)")
    print("X: ", "+"if (x_translational_flow_component>=0) else '-', "\tY: ","+"if (y_translational_flow_component>=0) else '-', "\tRotational: ", rotational_flow_magnitude,"(CCW)" if(rotational_flow_magnitude>=0) else "(CW)")

    


def detect_motion(frame1, frame2):
    # Resizing frames
    scale = 0.5
    frame1_resized = cv2.resize(frame1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    frame2_resized = cv2.resize(frame2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # Detecting flow using PCA
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
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

    net_grid = [[compute_grid_flow(grids_mag[i][j], grids_direction[i][j]) for j in range(3)] for i in range(3)]
    net_translation_flow_vector, rotation_flow_magnitude =compute_net_flow(net_grid)

    display_flow(net_translation_flow_vector, rotation_flow_magnitude)
    return (net_translation_flow_vector, rotation_flow_magnitude )

import cv2 as cv

cap = cv2.VideoCapture(2) # change according to camera input
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

    value=detect_motion(frame1, frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    frame1 = frame2.copy()  # Ensure frame1 is correctly updated

cap.release()
cv2.destroyAllWindows()