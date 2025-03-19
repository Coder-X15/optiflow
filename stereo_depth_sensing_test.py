import cv2 as cv
import numpy as np

def generate_depth_frame(left, right):
    # Read the frames
    left = cv.imread(left)
    right = cv.imread(right)

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
    num_disparities = 16*8  # Must be a multiple of 16
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

    # Apply a colormap to enhance visibility
    disparity_colormap = cv.applyColorMap(disparity_visual, cv.COLORMAP_JET)

    cv.imshow("Filtered Depth Map", disparity_colormap)
    cv.imwrite("filtered_depth_map.jpg", disparity_colormap)
    
    return disparity_colormap

left_image_path = 'l1.png'
right_image_path = 'r1.png'

disparity_map = generate_depth_frame(left_image_path, right_image_path)
cv.imshow('Disparity Map', cv.GaussianBlur(disparity_map, (5, 5), 0))
cv.waitKey(0)
cv.destroyAllWindows()
