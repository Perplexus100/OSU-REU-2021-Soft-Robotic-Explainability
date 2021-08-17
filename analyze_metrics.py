"""
Lucy Griswold
OSU REU "Robotics in the Real World" Summer 2021
A script to analyze visual grasp quality metrics from points generated from a 
labelled grasp image. 
"""
#used in is_regular()
import numpy as np

#used in angle()
import math

from shapely.geometry import Polygon

import sys


"""
Helper functions begin here
"""
def centroid(points):
    """
    Takes list of coordinate tuples that form an arbitrary polygon and returns 
    the centroid of that polygon as a tuple (x, y)
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = len(points)
    
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    
    return (centroid_x, centroid_y)


def distance(p1, p2):
    """
    Takes two coordinate tuples (x, y) and returns the distance between them
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle(x1, y1, x2, y2):
    """
    Takes the x and y coordinates of two vectors and uses the dot product to 
    calculate the angle between them. 
    Always returns an angle between 0 and 180 degrees.
    """
    if ((x1 == 0) and (y1 == 0)) or ((x2 == 0) and (y2 == 0)):
        return "Error: Divide by 0"
    
    else:
        numerator = ((x1 * x2) + (y1 * y2))
        denominator = math.sqrt((x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2))
        
        #returns angle in DEGREES
        return math.degrees((math.acos(numerator / denominator)))
    
    
def add_to_file(file_name, data):
    """
    Writes contact point array to a CSV file. Returns nothing.
    """
    with open('/Users/lucymore/OSU_REU/CSV_files/' + file_name + '.csv', 'a') as f:
        writer = csv.writer(f)

        # write multiple rows
        writer.writerows(data)
    
    
"""
Metric calculation functions begin here
"""
def is_regular(points):
    """
    Takes a list of coordinate tuples that form an arbitrary polygon and 
    compares the internal angles of that polygon to a normal polygon of the 
    corresponding number of sides. Returns score from 0 (completely unlike) 
    to 1 (exactly alike). Metric is float rounded to 4 units of precision. 
    """
    #number of vertices
    n = len(points)
    
    #calculates ideal angle in DEGREES
    norm_theta = ((n - 2) * 180) / n
    
    #creates list of internal angles
    angle_list = []

    for i in range(len(points)):
        
        p1 = points[i]
        ref = points[i - 1]
        p2 = points[i - 2]
        
        x1, y1 = p1[0] - ref[0], p1[1] - ref[1]
        x2, y2 = p2[0] - ref[0], p2[1] - ref[1]
        #print('Points', p1, ref, p2)
        #print('Angle', angle(x1, y1, x2, y2))
        
        angle_list.append( angle(x1, y1, x2, y2) )
    
    #divides list of internal angles by ideal angle
    ratio_list = []

    for angle_ in angle_list:
        ratio_list.append(angle_ / norm_theta)
    
    #averages list of ratios to produce final quality metric
    metric = np.mean(ratio_list)
    
    return round(metric, 4)


def is_distant(points, CoM):
    """
    Calculates normalized distance metric given list of contact points and 
    object's center of mass. Metric is normalized to a value between 0 and 1, 
    1 being no difference at all (ideal grasp). Output is float rounded to 4 
    units of precision. 
    """
    
    #calculates center of the grasp polygon
    poly_center = centroid(points)
    
    #distance between object CoM and polygon center
    difference = distance(poly_center, CoM)
    
    #calculates maximum distance between CoM and polygon vertices
    distance_list = []
    
    for i in range(len(points)):
        distance_list.append(distance(points[i], CoM))
        
    distance_max = max(distance_list)
    
    metric = 1 - (difference / distance_max)
    
    return round(metric, 4)


def is_area(points, area_max = 1):
    """
    Calculates the area of the grasp polygon given a list of contact points.
    
    OPTIONAL: area_max represents maximum span of the gripper when fully
    extended. Defaults to 1. In order to normalize metric, area_max MUST be
    inputted in correct units (pixels?)
    """
    
    poly = Polygon(points)
    area = poly.area
    
    metric = area / area_max
    
    return round(metric, 4)

    
def main():
    square = [
        ( 1.0,  1.0),
        ( 0.0,  0.0),
        ( 0.0,  1.0),
        ( 1.0,  0.0)
    ]
    
    octagon = [
        (2.992, -1.21),
        (1.21, -2.992),
        (-1.21, -2.992),
        (-2.992, -1.21),
        (-2.922, 1.21),
        (-1.21, 2.922),
        (1.21, 2.922),
        (2.922, 1.21)
    ]
    
    
    file_name = sys.argv[1]
   
    
    #Extracts point data from csv file written by click_detection_2.py
    points = np.loadtxt('/Users/lucymore/OSU_REU/CSV_files/' + file_name + '.csv', delimiter = ',')
    
    #center of object
    CoM = points[3]
    
    #center of gripper
    CoG = points[4]
    
    print("####")
    print("File Name:", file_name, "\n")
    print('Regularity Metric (0-1):', is_regular(points))
    
    print('Distance Metric (0-1):', is_distant(points, CoM))
    
    print('Area:', is_area(points))
    print("####")

if __name__ == '__main__':
    main() 
