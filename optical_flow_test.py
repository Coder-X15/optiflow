import cv2
import numpy as np

def compute_optical_flow(frame1, frame2, scale=0.5, blur_ksize=(5,5), method="farneback"):
    # Resize images
    frame1_resized = cv2.resize(frame1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    frame2_resized = cv2.resize(frame2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur (optional)
    if blur_ksize:
        gray1 = cv2.GaussianBlur(gray1, blur_ksize, 0)
        gray2 = cv2.GaussianBlur(gray2, blur_ksize, 0)

    # Compute optical flow
    if method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow  # Returns a NumPy array of shape (H, W, 2)

    elif method == "lucas-kanade":
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
        return np.hstack((p0, p1, st))  # Returns a NumPy array containing tracked points

# Load frames
frame1 = cv2.imread("img4.png")
frame2 = cv2.imread("img5.png")

# Check if images are loaded correctly
if frame1 is None or frame2 is None:
    print("Error: Could not read input images. Check file paths.")
    exit()

# Compute Optical Flow
flow = compute_optical_flow(frame1, frame2, method="farneback")

# Convert flow array into a visualization
h, w = flow.shape[:2]
flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# Normalize flow magnitude
flow_magnitude = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
flow_magnitude = flow_magnitude.astype(np.uint8)

# Apply color map to visualize the flow
flow_colormap = cv2.applyColorMap(flow_magnitude, cv2.COLORMAP_JET)

# Show result
cv2.imshow("Optical Flow Visualization", flow_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Optical Flow Computed and Visualized!")







    

