import numpy as np
from utils import *
import pickle
import copy
import random 


def strat_all_fucntion(a , b,i ,  save_path ="results/ShoesDataset" ): 
    p1 , p2 = strategy1(a[: , :2] , b[: , :2])
    get_interpolation_with_plots(p1 , p2 , "Reducing points in sketch with more points" , save_path +"//strategy1//{}.png".format(i))
    # Increasing points in sketch with more points
    p1 , p2 , = strategy2(a[: , :2] , b[: , :2])
    get_interpolation_with_plots(p1 , p2 , "Increasing randomly point in sketch with less pints by simple duplication" , save_path +"//strategy2//{}.png".format(i))

    # Increasing both by bresenham and sampling upto some max points 
    max_points = 500

    p1 , p2 = strategy3(a , b , max_points)
    get_interpolation_with_plots(p1 , p2 , "Bresenham on both sketches and sampling upto max points on both" ,save_path +"//strategy3//{}.png".format(i))

    # Increasing the one with lower number of points 
    p1 , p2 = strategy4(a , b)
    get_interpolation_with_plots(p1 , p2 , "Bresenham on sketch with lesser number of points" ,save_path +"//strategy4//{}.png".format(i))







file = open("ShoeV2_Coordinate" , 'rb')
object_file = pickle.load(file)
file.close()

file_list = list(object_file.keys())
#new_remove_list = []
#for sketch in range(len(file_list)):
 #   lngth = len(object_file[file_list[sketch]])
#    if lngth <= 500:
  #      new_remove_list.append(file_list[sketch])

random.shuffle(file_list)
sketch1_list = file_list[:200]
sketch2_list = file_list[200:400]
#random.shuffle(new_remove_list)
#sketch1_list = new_remove_list[:200]
#sketch2_list = new_remove_list[200:400]

for i in range(len(sketch1_list)):
    a = np.array(object_file[sketch1_list[i]])
    inter = copy.deepcopy(a)
    inter[: , 0] = 256
    inter[: , 1] = 256
    a[: , 2] = 0 
    a = inter - a
    del inter

    b = np.array(object_file[sketch2_list[i]])
    inter = copy.deepcopy(b)
    inter[: , 0] = 256
    inter[: , 1] = 256
    b[: , 2] = 0
    b = inter - b 
    del inter
    save_string= sketch1_list[i].split("/")[-1] +"_" + sketch2_list[i].split("/")[-1]
    strat_all_fucntion(a , b, save_string )
    print("Done with {}".format(i))
 


