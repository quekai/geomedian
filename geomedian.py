import numpy as np
import math
from math import pi
import utils
from externals import nputils

f = 1/298.257223
R = 6378137.0
R_km = R/1000
e = 8.1819190842622e-2
b = np.sqrt(np.power(R,2) * (1-np.power(e,2)))
ep = np.sqrt((np.power(R,2)-np.power(b,2))/np.power(b,2))


def llaToECEF(points):
    lat = np.radians(points[:,0])
    lon = np.radians(points[:,1])
    if points.shape[1] == 2:
        h = np.zeros((points.shape[0],), dtype=np.float64)
    else:
        h = np.radians(points[:,2])

    lambds = np.arctan((1-f)**2*np.tan(lat))
    rs = np.sqrt((R**2)/(1+(1/(1-f)**2-1)*np.sin(lambds)**2))

    x = rs * np.cos(lambds) * np.cos(lon) + h * np.cos(lat)*np.cos(lon)
    y = rs * np.cos(lambds) * np.sin(lon) + h * np.cos(lat)*np.cos(lon)
    z = rs * np.sin(lambds) + h * np.sin(lat)

    return np.stack((x, y, z), axis=1)

def ECEFTolla(points):
    p = np.sqrt(np.power(points[:,0], 2)+np.power(points[:,1],2))
    th = np.arctan2(R*points[:,2], b*p)
    
    lon = np.arctan2(points[:,1], points[:,0])
    lat = np.arctan2((points[:,2]+ep*ep*b*np.power(np.sin(th),3)),\
        (p-e*e*R*np.power(np.cos(th),3)))
    n = R/np.sqrt(1-e*e*np.power(np.sin(lat),2))
    alt = p/np.cos(lat)-n
    lat = (lat*180)/pi
    lon = (lon*180)/pi
    
    return np.stack((lat, lon, alt), axis=1)

def dist_array(points, ref):
    return np.sqrt(
        (ref[0] - points[:,0])**2 +
        (ref[1] - points[:,1])**2 +
        (ref[2] - points[:,2])**2)

def denomsum(distances):
    return np.sum(1.0/distances)

# from: http://stackoverflow.com/questions/14766194/testing-whether-a-numpy-array-contains-a-given-row
def row_in_array(row, array):
    return any((array[:]==row).all(1))

def median(points, threshold=0.1, ECEF = False):
    # if all the points are the same, skip the procedure
    if nputils.array.unique_rows(points).shape[0] == 1:
        return points[0,:]
    if ECEF:
        card_points = points
    else:
        card_points = llaToECEF(points)
    temp_median = np.mean(card_points, axis=0)
    while row_in_array(temp_median, card_points):
        temp_median += 0.01

    max_iter = 100
    last_obj = 0.0
    for x in range(max_iter):
        distances = dist_array(card_points, temp_median)
        distances[distances==0] = threshold/100.0
        current_obj = np.sum(distances)
        if math.fabs(last_obj-current_obj) < threshold:
            break
        last_obj = current_obj
        denom = denomsum(distances)
        nextx = np.sum(card_points[:,0]/distances)/denom
        nexty = np.sum(card_points[:,1]/distances)/denom
        nextz = np.sum(card_points[:,2]/distances)/denom
        temp_median = [nextx, nexty, nextz]
    temp_median = np.array([temp_median])
    if ECEF:
        return temp_median[0]
    else:
        result = ECEFTolla(temp_median)[0]
        # depending on whether two or three dimensional coordinates were given
        return result[:points.shape[1]] 

       
