"""
This python script uses OpenCV to detect clicks on an image and produces a list of the coordinates of each click. Name of image file should be passed as command line argument. 
"""

import cv2
import numpy as np
import csv
import sys

import pdb

### GLOBAL VARIABLES ###

#click counter
counter = 0

#Initiates array of contact points
# Order for metrics analysis is: [pt, pt, pt, center of object, cen. of gripper]
# Colors: white, white, white, green, red
# will throw an error when number of clicks exceeds array dimension 
contacts = np.zeros((5,2), np.int)

#allows image to be re-written when clicked to draw circles
global_img = None

### ### ###

def loadImage(img_number):
    """
    Takes an image name as a string, resizes it, and opens the image in a new window. Prints original and resized dimensions. Also calls mousePoints through setMouseCallback each time a mouse event occurs within the image window. Returns nothing. 
    """
    
    path = '/Users/lucymore/OSU_REU/Grasp_images/' + img_number + '.jpeg'
    
    img = cv2.imread(path)

    print('Original Dimensions : ',img.shape)

    #percent of original size new image should be
    scale_percent = 15
    
    #Size information for new image
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    #Resizes image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
    print('Resized Dimensions : ',resized.shape)
 
    global global_img
    global_img = resized
    cv2.imshow("Resized image", resized)
    
    cv2.setMouseCallback("Resized image", mousePoints)
    
    
def mousePoints(event, x, y, flags, params):
    """
    Records the x and y coordinates of a left-button mouse click within an open image window to the array of contact points. Prints click counter and updated contacts list. Returns nothing. 
    """
    global counter  
    global global_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
    
        #adds mouse click coordinates to array
        contacts[counter] = x,y
        print(contacts)
        
        counter += 1
        print(counter)
        print("###")
    
        #Creates circles on clicks 
        
        #Center point of click
        center_pt = np.array([x, y])
        
        #contact points = white
        if counter <= 3:
            cv2.circle(global_img, center_pt, 3, (255, 255, 255), cv2.FILLED)
        
        #center of object = green 
        elif counter == 4:
            cv2.circle(global_img, center_pt, 3, (0, 255, 0), cv2.FILLED)
        
        #center of gripper = red
        elif counter == 5:
            cv2.circle(global_img, center_pt, 3, (0, 0, 255), cv2.FILLED)
            
        cv2.imshow("Resized image", global_img)
        
        
def writeToFile(file_name):
    """
    Writes contact point array to a CSV file. Returns nothing.
    """
    with open('/Users/lucymore/OSU_REU/CSV_files/' + file_name + '.csv', 'w') as f:
        writer = csv.writer(f)

        # write multiple rows
        writer.writerows(contacts)
    

def main():
    #loads image with same name as command line arg
    loadImage(sys.argv[1])
    
    #waits for any key to be pressed, then destroys the image window
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    #CSV filename will be the same as image file being processed (command line arg)
    writeToFile(sys.argv[1])
    
    
if __name__ == '__main__':
    main() 