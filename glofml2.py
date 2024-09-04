import cv2
import numpy as np
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime


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

def load_data_from_db():
    """
    Load image paths, dates, and area targets from MySQL database.
    """
    # Connect to the MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="your_database_name"
    )
    cursor = connection.cursor()

    # Query the database to get image paths, dates, and area targets
    cursor.execute("SELECT image_path, date, area FROM your_table_name")
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    # Process the results
    image_paths = []
    dates = []
    area_targets = []
    
    for row in results:
        image_paths.append(row[0])
        dates.append(datetime.strptime(row[1], "%Y-%m-%d"))
        area_targets.append(row[2])
    
    return image_paths, dates, area_targets

def calculate_time_index(dates):
    """
    Calculate the time index as the number of days from the first image date.
    """
    start_date = dates[0]
    time_indices = [(date - start_date).days for date in dates]
    return time_indices

def main():
    # Load data from the database
    image_paths, dates, area_targets = load_data_from_db()

    # Calculate time indices
    time_indices = calculate_time_index(dates)

    # Extract features and prepare data for model training
    X = []
    y = []
    
    for i, (image_path, time_index, area_target) in enumerate(zip(image_paths, time_indices, area_targets)):
        features = extract_features(image_path, time_index)
        X.append(features)
        y.append(area_target)
    
    X = np.array(X)
    y = np.array(y)

    # Split the data (keep the last image as test data)
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict area for the latest image (the test image)
    y_pred_test = model.predict([X[-1]])
    actual_test = y[-1]

    # Calculate the percentage difference
    percentage_diff = abs((y_pred_test[0] - actual_test) / actual_test) * 100

    # Print the predicted, actual area, and percentage difference for the latest image
    print(f"Test Image (Latest): Predicted Area = {y_pred_test[0]:.2f} sq. meters")
    print(f"Actual Area = {actual_test} sq. meters")
    print(f"Percentage Difference = {percentage_diff:.2f}%")

    # Calculate Mean Squared Error for the test set
    y_test_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    print(f"Mean Squared Error on Test Set: {mse:.2f}")

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

if __name__ == "__main__":
    main()
