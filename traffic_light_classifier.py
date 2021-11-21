import cv2 # computer vision library

import helpers # helper functions
import test_functions # test functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images



def vizualize_input(image_list):
    yellow_index=730
    red_index=0
    green_index=783

    selected_image_red = IMAGE_LIST[red_index][0]
    selected_label_red = IMAGE_LIST[red_index][1]

    selected_image_yellow = IMAGE_LIST[yellow_index][0]
    selected_label_yellow = IMAGE_LIST[yellow_index][1]

    selected_image_green = IMAGE_LIST[green_index][0]
    selected_label_green = IMAGE_LIST[green_index][1]

    # Visualize the individual color channels
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    ax1.set_title('red light')
    print('Image shape: ', selected_image_red.shape)
    print('Image label: ', selected_label_red)
    ax1.imshow(selected_image_red)

    ax2.set_title('yellow light')
    print('Image shape: ', selected_image_yellow.shape)
    print('Image label: ', selected_label_yellow)
    ax2.imshow(selected_image_yellow)

    ax3.set_title('green light')
    print('Image shape: ', selected_image_green.shape)
    print('Image label: ', selected_label_green)
    ax3.imshow(selected_image_green)
    
    return True
    
def standardize_input(image):
    """ This function takes in an RGB image and return a new, standardized version 
        by resizing the image so that all "standard" images are the same size (32x32) 
    """    
    standard_image = np.copy(image) 
    # Use OpenCV's resize function
    standard_image = cv2.resize(standard_im, (32, 32))    
    return standard_image

def one_hot_encode(label): 
    " Given a label - red, green, or yellow - return a one-hot encoded label"
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
        # Append the image and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))        
    return standard_list


def create_feature(rgb_image):
    """ Create a brightness feature that takes in an RGB image and outputs a feature vector
        using HSV colorspace values """
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)   
    # Create and return a feature vector
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
    " to help accurately label the traffic light images "
    feature=create_feature(rgb_image)
    maxBrightness=np.argmax(feature)
    return maxBrightness  
  

def estimate_label(rgb_image):   
    """ This function takes in an RGB image as input, analyze that image 
    using the feature creation code and output a one-hot encoded label """   
    predicted_label = []
    #Extract feature(s) from the RGB image
    maxBrightness=improve_feature(rgb_image)
    if maxBrightness==0:
        label="red"
    elif maxBrightness==1:
        label="yellow"
    elif maxBrightness==2:
        label="green"   
    #output a one-hot encoded label
    predicted_label_ohl=one_hot_encode(label)       
    return predicted_label_ohl
         
def get_misclassified_images(test_images):
    " Constructs a list of misclassified images given a list of test images and their labels "
    
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []
    # Iterate through all the test images
    # Classify each image and compare to the true label
    for test_image in test_images:
        # Get true data
        image = test_image[0]
        true_label = test_image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."       
        # Get predicted label from your classifier
        predicted_label = estimate_label(image)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."
        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((image, predicted_label, true_label))            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

# Load training data
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
# Display the image, shape and label
vizualize_input(IMAGE_LIST)

# Image data directories
IMAGE_DIR_TEST = "traffic_light_images/test/"
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)
# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Check criteria 1
# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total
print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))
# Display an image in the `MISCLASSIFIED` list 
misclassified_image=MISCLASSIFIED[13][0]
plt.imshow(misclassified_image)
# Print out its predicted label - to see what the image *was* incorrectly classified as
misclassified_image_label=MISCLASSIFIED[0][1]
print("misclassified_image_label", misclassified_image_label)

# Check criteria 2
tests = test_functions.Tests()
if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")
