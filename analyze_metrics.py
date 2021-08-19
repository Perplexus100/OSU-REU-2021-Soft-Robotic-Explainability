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
import csv


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
    Writes contact point array to an existing CSV file. Returns nothing.
    """


    with open('/Users/lucymore/OSU_REU/CSV_files/' + file_name + '.csv', 'a') as f:
        writer = csv.writer(f, delimiter = ' ')

        writer.writerow([str(data)])


def add_row_to_csv(file_name, list_of_row):
        """
        Adds a row of objects to a csv. This is relying on the str() function to
        work on the objects
        """

        # create the string of the row
        row_str = ''

        for i, elem in enumerate(list_of_row):
            if i != 0:
                row_str += ','

            row_str += str(elem)

        print(row_str)
        with open('/Users/lucymore/OSU_REU/CSV_files/' + file_name + '.csv', 'a') as f:
            f.write(row_str)



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

        angle_list.append( angle(x1, y1, x2, y2) )


    #sum of the differences between internal angles when polygon degenerates to a line
    theta_max = ((n - 2)*(180 - norm_theta)) + (2 * norm_theta)

    #sums together the absolute differences between internal angle and ideal angle
    difference_sum = 0

    for angle_ in angle_list:
        difference_sum += abs(angle_ - norm_theta)

    #normalizes metric to between 0 and 1, 1 being the best value
    metric = 1 - (1 / theta_max) * difference_sum

    return round(metric, 4)


def is_distant(points, CoM):
    """
    The distance metric is defined as the distance between the centroid of
    the grasp polygon and the object's center of mass.

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

    triangle = [
        (1, 1),
        (1, 0),
        (2, 0)
    ]

    file_name = sys.argv[1]

    #Extracts point data from csv file written by click_detection_2.py
    #input format is [point, point, point, CoM, gripper center]
    points = np.loadtxt('/Users/lucymore/OSU_REU/CSV_files/' + file_name + '.csv', delimiter = ',')

    #contact polygon
    contacts = points[:3]

    #center of object
    CoM = points[3]

    #center of gripper
    CoG = points[4]

    print("####")
    print("\nFile Name:", file_name, "\n")
    print('Regularity Metric (0-1):', is_regular(contacts))

    print('Distance Metric (0-1):', is_distant(contacts, CoM))

    print('Area:', is_area(contacts))
    print("\n####")


    add_row_to_csv('metric_data',   [file_name, \
                                    is_regular(contacts), \
                                    is_distant(contacts, CoM), \
                                    is_area(contacts)])

    # add_to_file('metric_data', str(file_name))
    # add_to_file('metric_data', is_regular(contacts))
    # add_to_file('metric_data', is_distant(contacts, CoM))
    # add_to_file('metric_data', is_area(contacts))


if __name__ == '__main__':
    main()
