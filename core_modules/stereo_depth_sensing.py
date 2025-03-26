import cv2 as cv
import numpy as np

def generate_depth_frame(left, right,cam_focal_length= 4.4e-3, baseline=25e-2):
    '''
    INPUT:

    Parameters:
        left: left camera feed as numpy array
        right: right camera feed as numpy array
        cam_focal_length:focal length of the camera used in the stereo cam module (in metres)
        baseline: relative separation of the cameras in the stereo-camera design (in metres)(default 25 cm (0.25 m))

    OUTPUT:
    Numpy array corresponding to the depth frame in metres.
        
    '''

    left_gray = cv.cvtColor(left, cv.COLOR_RGB2GRAY)
    right_gray = cv.cvtColor(right, cv.COLOR_RGB2GRAY)

    # Apply Gaussian Blur
    imgL = cv.GaussianBlur(left_gray, (5, 5), 0)
    imgR = cv.GaussianBlur(right_gray, (5, 5), 0)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    imgL = clahe.apply(imgL)
    imgR = clahe.apply(imgR)

    # StereoSGBM Parameters
    min_disparity = 5
    num_disparities = 16*4  # Must be a multiple of 16
    block_size = 3  # Should be an odd number
    p1 = 8 * block_size**2  # Smoothness parameter 1
    p2 = 32 * block_size**2  # Smoothness parameter 2
    disp12_max_diff = 3
    uniqueness_ratio = 10
    speckle_window_size = 50
    speckle_range = 32
    pre_filter_cap = 63

    # StereoSGBM Matching
    stereo = cv.StereoSGBM_create(
        numDisparities=num_disparities,
        blockSize=block_size,
        minDisparity=min_disparity,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=pre_filter_cap,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # WLS Filtering
    lmbda = 8000
    sigma = 1.2
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    right_matcher = cv.ximgproc.createRightMatcher(stereo)
    disparity_right = right_matcher.compute(imgR, imgL).astype(np.float32) / 16.0
    filtered_disparity = wls_filter.filter(disparity, imgL, None, disparity_right)

    # Normalize for visualization
    disparity_visual = cv.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    disparity_visual = np.uint8(disparity_visual)

    # Avoid division by zero or negative disparities
    valid_disparity = np.where(filtered_disparity > 0, filtered_disparity, 1e-6)
    depth_map = (cam_focal_length * baseline) / valid_disparity

    #defining region of interest by slicing
    depth_map=depth_map[:,70:: ]

    return depth_map
