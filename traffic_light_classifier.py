import cv2 # computer vision library

import helpers # helper functions
import test_functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

%matplotlib inline

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):   
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image) 
    # Use OpenCV's resize function
    standardized_im = cv2.resize(standard_im, (32, 32))    
    return standardized_im

## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label
def one_hot_encode(label):    
    dic_standard={'red':[1,0,0],'yellow':[0,1,0],'green':[0,0,1]}
    one_hot_encoded=dic_standard[label]
    return one_hot_encoded

def standardize(image_list):    
    # Empty image data array
    standard_list = []
    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        # Standardize the image
        standardized_im = standardize_input(image)
        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    
        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))        
    return standard_list

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image):
    
    #convert your image to HSV
    ## TODO: Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)   
    ## TODO: Create and return a feature value and/or vector
    feature = []
    #keep only the v-channel
    v = hsv[:,:,2]    
    # Region identification
    red_Region    = v[:11, 4:24]
    yellow_Region = v[11:21, 4:24]
    green_Region  = v[21:, 4:24]   
    regions = [red_Region, yellow_Region, green_Region]
    # Calculate the average brightness for each region
    avg_brightness = [np.average(region) for region in regions]
    return avg_brightness
  
  def improve_feature(rgb_image):
    feature=create_feature(rgb_image)
    maxBrightness=np.argmax(feature)
    return maxBrightness  
  
# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):   
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = []
    maxBrightness=improve_feature(rgb_image)
    if maxBrightness==0:
        label="red"
    elif maxBrightness==1:
        label="yellow"
    elif maxBrightness==2:
        label="green"
    predicted_label_ohl=one_hot_encode(label)    
    return predicted_label_ohl
         
# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []
    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:
        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."       
        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."
        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels



# Image data directories
IMAGE_DIR_TEST = "traffic_light_images/test/"
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)
# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))
# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
misclassified_image=MISCLASSIFIED[13][0]
plt.imshow(misclassified_image)
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
misclassified_image_label=MISCLASSIFIED[0][1]
print("misclassified_image_label", misclassified_image_label)
