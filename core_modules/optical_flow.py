import cv2
import math
import numpy as np

def compute_grid_flow(flow_magnitude, flow_direction,frame_width, frame_height, altitude, frame_rate,fov_x=45.83*math.pi/180, fov_y=70*math.pi/180):
    ''' INPUT: a 2D flow_magnitude frame  where each cell corresponds to the magnitude of flow vector at the corresponding pixel\t,\n
                another 2D flow_direction frame where each cell corresponds to the angle of flow vector  with repect to x axis at the corresponding pixel\t,\n
                horizontal_fov of stereocamera, vertical_fov of stereocamera, altitude and frame rate
        OUTPUT: a tuple of the form (velocity_magnitude,direction_angle) of the resultant velocity vector of all flow vectors in the frame'''
    V_grid_x = 0
    V_grid_y = 0

    for i in range(flow_magnitude.shape[0]):
        for j in range(flow_magnitude.shape[1]):
            V_grid_x += (2*(flow_magnitude[i, j]) * math.cos(flow_direction[i, j])*frame_rate*altitude*math.tan(fov_x/2))/frame_width
            V_grid_y += (2*((flow_magnitude[i, j]) * math.sin(flow_direction[i, j]) * -1)*frame_rate*altitude*math.tan(fov_y/2))/frame_height

    V_grid_x/=(flow_magnitude.shape[0]*flow_magnitude.shape[1])
    V_grid_y/=(flow_magnitude.shape[0]*flow_magnitude.shape[1])
    return (math.sqrt(V_grid_x**2 + V_grid_y**2),(2*math.pi)- math.atan2(V_grid_y,V_grid_x))


def vector_component(vector1, vector2):
    '''utility method to compute component of vector1 in the direction of vector2'''
    v1_mag, v1_angle = vector1
    v2_mag, v2_angle = vector2
    return v1_mag * math.cos(v2_angle - v1_angle)

def compute_net_flow(grids, fov_x, fov_y,frame_width, frame_height, altitude):
    '''
    INPUT: 3 x 3 array of (flow_magnitude, flow_angle) tuples.
    OUTPUT: a tuple of net_translation_velocity in the format (magnitude, angle) and net angular velocity (omega).
    '''
    net_V_x = 0
    net_V_y = 0

    for row in range(len(grids)):
        for col in range(len(grids[row])):
            net_V_x += grids[row][col][0] * math.cos(grids[row][col][1])
            net_V_y += grids[row][col][0] * math.sin(grids[row][col][1]) * -1  # Negating y for correct flow

    net_V_x/=9
    net_V_y/=9


    # Computing net translational velocity
    net_translation_velocity = (
        math.sqrt(net_V_x**2 + net_V_y**2),
        (2 * math.pi) - math.atan2(net_V_y, net_V_x)
    )

    # Computing net angular velocity
    principal_vector = (1, math.pi)
    next_steps = ['l', 'd', 'r', 'u']
    next_step_index = 0
    next_step = 'l'
    curr_pos = [0, 1]
    net_omega = 0

    for _ in range(8):  #stops within the 3x3 grid

        #Adding omega component
        if(math.abs(vector_component(principal_vector, (1,0)))==1):
            #the principal vector is parallel or antiparallel to x axis
            r = ((2*math.tan(fov_y/2)*altitude)/3)
            net_omega += (vector_component(grids[curr_pos[0]][curr_pos[1]], principal_vector)/r)
        elif(math.abs(vector_component(principal_vector, (1,0)))==0):
            #the principal vector is orthogonal to x axis 
            r = ((2*math.tan(fov_x/2)*altitude)/3)
            net_omega += (vector_component(grids[curr_pos[0]][curr_pos[1]], principal_vector)/r)
        else:
            #the principal vector is at an net angle of pi/4 with respect to the parallel or antiparallel direction of x axis (CORNER GRIDS)
            r = math.sqrt( ((2*math.tan(fov_x/2)*altitude)/3)**2    +  ((2*math.tan(fov_y/2)*altitude)/3)**2)
            net_omega += (vector_component(grids[curr_pos[0]][curr_pos[1]], principal_vector)/r)

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

        # Updating principal_vector (rotating by +pi/4 CCW)
        principal_vector = (principal_vector[0], principal_vector[1] + (math.pi / 4))

    return (net_translation_velocity, (net_omega/8))



# def display_flow(translational_flow_vector_polar,rotational_flow_magnitude ):
#     '''display method for debugging'''

#     x_translational_flow_component=translational_flow_vector_polar[0]*math.cos(translational_flow_vector_polar[1])
#     y_translational_flow_component=translational_flow_vector_polar[0]*math.sin(translational_flow_vector_polar[1])*-1

#     # print("X: ", x_translational_flow_component, "\tY: ", y_translational_flow_component, "\tRotational: ", rotational_flow_magnitude,"(CCW)" if(rotational_flow_magnitude>=0) else "(CW)")
#     print("X: ", "+"if (x_translational_flow_component>=0) else '-', "\tY: ","+"if (y_translational_flow_component>=0) else '-', "\tRotational: ", rotational_flow_magnitude,"(CCW)" if(rotational_flow_magnitude>=0) else "(CW)")

    


def detect_motion(depth_frame1, depth_frame2, altitude, frame_rate, h_fov=45.83*math.pi/180, v_fov=70*math.pi/180):

    #determining frame_height and frame_width
    frame_height=depth_frame1.shape[1]
    frame_width = depth_frame1.shape[0]

    #Normalizing the depth values
    depth_frame1 = cv2.normalize(depth_frame1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_frame2 = cv2.normalize(depth_frame2, None, 0 , 255, cv2.NORM_MINMAX).astype(np.uint8)
 
    #Applying Colourmaps to frames to better visibility
    frame1 = cv2.applyColorMap(depth_frame1, cv2.COLORMAP_JET)
    frame2 = cv2.applyColorMap(depth_frame2, cv2.COLORMAP_JET)

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

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

    net_grid = [[compute_grid_flow(grids_mag[i][j], grids_direction[i][j],h_fov, v_fov,frame_width, frame_height,altitude, frame_rate) for j in range(3)] for i in range(3)]
    net_translation_flow_vector, angular_velocity =compute_net_flow(net_grid)

    return (net_translation_flow_vector, angular_velocity)