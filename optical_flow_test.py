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
    # if blur_ksize:
    #     gray1 = cv2.GaussianBlur(gray1, blur_ksize, 0)
    #     gray2 = cv2.GaussianBlur(gray2, blur_ksize, 0)

    # Compute optical flow
    if method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow  # Returns a NumPy array of shape (H, W, 2)

    # elif method == "lucas-kanade":
    #     feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    #     lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    #     p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    #     return np.hstack((p0, p1, st))  # Returns a NumPy array containing tracked points

    elif method == "PCA":
        pca_flow = cv2.optflow.createOptFlow_PCAFlow()  # Corrected instantiation
        flow = pca_flow.calc(gray1, gray2, None)  # Corrected function usage
        return flow  # Returns (H, W, 2) optical flow

def robust_normalization(flow_magnitude, lower_percentile=5, upper_percentile=95):
    min_val = np.percentile(flow_magnitude, lower_percentile)
    max_val = np.percentile(flow_magnitude, upper_percentile)
    flow_magnitude = np.clip(flow_magnitude, min_val, max_val)  # Remove outliers
    flow_magnitude = (flow_magnitude - min_val) / (max_val - min_val)  # Normalize to [0,1]
    flow_magnitude = (flow_magnitude * 255).astype(np.uint8)  # Convert to image format
    return flow_magnitude

# Open webcam
cap = cv2.VideoCapture(0) # change according to camera input
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
    flow1= compute_optical_flow(frame1, frame2, method="PCA")
    flow2= compute_optical_flow(frame1, frame2, method="farneback")

    # Convert flow array into a visualization
    h1, w1 = flow1.shape[:2]
    flow1_magnitude, flow_angle = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
    h2, w2 = flow2.shape[:2]
    flow2_magnitude, flow_angle = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

    # Normalize flow magnitude
    flow1_magnitude=robust_normalization(flow1_magnitude)
    flow2_magnitude=robust_normalization(flow2_magnitude)

    flow1_magnitude = flow1_magnitude.astype(np.uint8)
    flow2_magnitude = flow2_magnitude.astype(np.uint8)
    # import scipy.ndimage  
    # flow1_magnitude = scipy.ndimage.median_filter(flow1_magnitude, size=3)
    # flow2_magnitude = scipy.ndimage.median_filter(flow2_magnitude, size=3)  


    # Apply color map to visualize the flow
    flow1_colormap = cv2.applyColorMap(flow1_magnitude, cv2.COLORMAP_JET)
    flow2_colormap = cv2.applyColorMap(flow2_magnitude, cv2.COLORMAP_JET)

    cv2.imshow("PCA optical flow", cv2.resize(flow1_colormap,(500, 500)))
    cv2.imshow("farenback optical flow", cv2.resize(flow2_colormap,(500, 500)))
    cv2.imshow("Input feed", cv2.resize(frame2, (500, 500)))

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    frame1 = frame2.copy()  # Ensure frame1 is correctly updated

cap.release()
cv2.destroyAllWindows()
