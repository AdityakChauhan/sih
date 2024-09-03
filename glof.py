import cv2
import numpy as np

def count_hue_range_pixels(image_path, hue_start=208, hue_end=212, saturation_min=0, saturation_max=255, lightness_min=0, lightness_max=255):
    """
    Count the number of pixels in an image that have a hue within the specified range.

    :param image_path: Path to the image file.
    :param hue_start: Starting value of the hue range.
    :param hue_end: Ending value of the hue range.
    :param saturation_min: Minimum saturation value for the pixels to be counted.
    :param saturation_max: Maximum saturation value for the pixels to be counted.
    :param lightness_min: Minimum lightness value for the pixels to be counted.
    :param lightness_max: Maximum lightness value for the pixels to be counted.
    :return: Number of pixels with hue in the specified range.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image format.")
    
    # Convert the image from BGR to HLS (Hue, Lightness, Saturation)
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Define the lower and upper bounds for the hue range
    lower_bound = np.array([hue_start // 2, lightness_min, saturation_min], dtype=np.uint8)
    upper_bound = np.array([hue_end // 2, lightness_max, saturation_max], dtype=np.uint8)

    # Create a mask that matches pixels within the specified hue range
    mask = cv2.inRange(image_hls, lower_bound, upper_bound)

    # Count the number of non-zero pixels in the mask
    pixel_count = np.count_nonzero(mask)

    return pixel_count

# Provide the path to your image
image_path = "glof_photo5.jpg"

# Count the pixels with hue in the range 200-220
total_pixel_count = count_hue_range_pixels(
    image_path, 
    hue_start=208, 
    hue_end=219, 
    saturation_min=0,  # Adjust these values based on the image's characteristics
    saturation_max=255, 
    lightness_min=0, 
    lightness_max=255
)
print(f"Total number of pixels with hue in the range 200-220: {total_pixel_count}")
spactial_resolution = 4
area = total_pixel_count*spactial_resolution
print("Total area in meter sq is: ", area)

total_depth = 0.104 * (area**0.42) #using Huggel's formula
print("Total Depth: ", total_depth)
# print("Total Water currently holding in MCM: ", water_hold/1000000)
water_hold = 0.104 * (area**1.42) #using Huggel's formula
print("Total Water currently holding in cubic meter: ", water_hold)
print("Total Water currently holding in MCM: ", water_hold/1000000)