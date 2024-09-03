import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def extract_features(image_path, time_index):
    """
    Extract features from the image. Count hue-range pixels and include a time factor.
    """
    # Load the image and convert to HLS
    image = cv2.imread(image_path)
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    # Define the hue range and create a mask
    hue_start, hue_end = 208, 212
    lower_bound = np.array([hue_start // 2, 0, 0], dtype=np.uint8)
    upper_bound = np.array([hue_end // 2, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(image_hls, lower_bound, upper_bound)
    
    # Count pixels in the hue range
    pixel_count = np.count_nonzero(mask)
    
    # Calculate area (or use other features)
    spatial_resolution = 4  # Example resolution in square meters per pixel
    area = pixel_count * spatial_resolution
    
    # Include the time factor
    features = np.array([area, time_index])
    
    return features

def load_data(image_paths, area_targets):
    """
    Load data and extract features for each image.
    """
    X = []
    y = []
    
    for i, (image_path, area_target) in enumerate(zip(image_paths, area_targets)):
        features = extract_features(image_path, i + 1)  # Using i+1 as a time factor
        X.append(features)
        y.append(area_target)
    
    return np.array(X), np.array(y)

# Example data: image paths and corresponding actual area targets (in square meters)
image_paths = ["glof_photo1.jpg", "glof_photo2.jpg", "glof_photo3.jpg", "glof_photo4.jpg", "glof_photo5.jpg"]
area_targets = [5000, 10000, 3500, 991336, 1021900]  # Example area values for Photos 1-3; Actual areas for Photos 4-5

# Load and split the data (keep the last image as test data)
X, y = load_data(image_paths, area_targets)
X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict area for the latest image (the test image)
y_pred_test = model.predict([X[-1]])
actual_test = y[-1]

# Print the predicted and actual area for the latest image
print(f"Test Image (Latest): Predicted Area = {y_pred_test[0]:.2f} sq. meters, Actual Area = {actual_test} sq. meters")

# Predict areas for all images (including the new ones)
predicted_areas = model.predict(X)

# Plotting the trend of expected lake area vs. actual lake area
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(image_paths) + 1), predicted_areas, label='Predicted Area', marker='o')
plt.plot(range(1, len(image_paths) + 1), area_targets, label='Actual Area', marker='o')
plt.title('Expected vs Actual Lake Area with Time')
plt.xlabel('Image Number (Chronological Order)')
plt.ylabel('Lake Area (sq. meters)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the volume and depth using Huggel's formula for actual data
for i, actual_area in enumerate(area_targets):
    depth = 0.104 * (actual_area ** 0.42)
    volume = 0.104 * (actual_area ** 1.42)
    print(f"\nPhoto {i + 1}:")
    print(f"Actual Area = {actual_area} sq. meters")
    print(f"Estimated Depth = {depth:.2f} meters")
    print(f"Estimated Volume = {volume:.2f} cubic meters")
    print(f"Estimated Volume = {volume / 1_000_000:.2f} MCM")
