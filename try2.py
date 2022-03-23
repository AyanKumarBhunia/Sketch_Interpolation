import numpy as np
import scipy.ndimage
from bresenham import bresenham
from PIL import Image
import copy 

def mydrawPNG(vector_image):
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    final_list = []

    for i in range( 0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        final_list.extend([list(i) for i in cordList])
        
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
        

    print(final_list)

    return final_list

def preprocess(sketch_points, side=256.0):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([side, side])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points



import pickle
from utils import visualize_numpy_points

file = open("ShoeV2_Coordinate" , 'rb')
object_file = pickle.load(file)
file.close()

a = list(object_file.keys())
print(a)
shoe1 = np.array(object_file[a[0]])
inter = copy.deepcopy(shoe1)
inter[: , 0] = 256
inter[: , 1] = 256
shoe1[: , 2] = 0
shoe1 = inter - shoe1
del inter

shoe1 = preprocess(shoe1)
shoe1 = mydrawPNG(shoe1)
visualize_numpy_points(shoe1, "g")