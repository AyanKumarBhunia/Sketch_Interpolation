import numpy as np
from utils import *
import matplotlib.pyplot as plt

#ant_path = "data\\ant.npy"
#aeroplane_path = "data\\airplane.npy"

#ant_data = open_numpy_file(ant_path)
#aeroplane_data = open_numpy_file(aeroplane_path)
#visualize_numpy_points(ant_data)
#visualize_numpy_points(aeroplane_data)

# Strategy1 - choosing the minimum of both and uniform sampling of the maximum 
#p1 , p2 = strategy1(ant_data , aeroplane_data)
#interpolated_points = get_interpolation(p1 , p2 , 0.9)
#visualize_numpy_points(p1 , "ant")
#visualize_numpy_points(p2 , "airplane") 
#visualize_numpy_points(interpolated_points , "interpolated point with lambda = 0.9")

#Strategy2 - paddiing it with 0 coordinates for the minimum points 
#p1 , p2 = strategy2(ant_data , aeroplane_data)
#interpolated_points = get_interpolation(p1 , p2 , 0.9)
#visualize_numpy_points(p1 , "ant")
#visualize_numpy_points(p2 , "airplane") 
#visualize_numpy_points(interpolated_points , "interpolated point with lambda = 0.9")
                                                      
                                        
import pickle
import copy

file = open("a" , 'rb')
object_file = pickle.load(file)
file.close()

a = np.array(object_file[list(object_file.keys())[12084]])
inter = copy.deepcopy(a)
inter[: , 0] = 800
inter[: , 1] = 800
a[: , 2] = 0 
a = inter - a
del inter

b = np.array(object_file[list(object_file.keys())[608]])
inter = copy.deepcopy(b)
inter[: , 0] = 800
inter[: , 1] = 800
b[: , 2] = 0
b = inter - b 
del inter

#visualize_numpy_points(b , "g") 
# Reducing points in sketch with more points
#p1 , p2 = strategy1(a[: , :2] , b[: , :2])
#get_interpolation_with_plots(p1 , p2 , "Reducing points in sketch with more points")
# Increasing points in sketch with more points
#p1 , p2 , = strategy2(a[: , :2] , b[: , :2])
#get_interpolation_with_plots(p1 , p2 , "Increasing randomly point in sketch with less pints by simple duplication")

# Increasing both by bresenham and sampling upto some max points 
max_points = 500

p1 , p2 = strategy3(a , b , max_points)
#get_interpolation_with_plots(p1 , p2 , "Bresenham on both sketches and sampling upto max points on both ")

# Increasing the one with lower number of points 
#p1 , p2 = strategy4(a , b)
#get_interpolation_with_plots(p1 , p2 , "Bresenham on sketch with lesser number of points")


#visualize_numpy_points(a , "h")
#visualize_numpy_points_two(np.array(p1))
#stroke_list = np.split(p1[:, :2], np.where(p1[:, 2])[0] + 1, axis=0)

stroke_map = {}
for point in p1:
    if point[2] not in stroke_map:
        stroke_map[point[2]] = [point[:2]]
    else:
        stroke_map[point[2]].append(point[:2])


x_count = 0
count = 0
for _ in stroke_map:
       stroke = np.array(stroke_map[_])
       #stroke_buffer = np.array(stroke[0])
       for x_num in range(len(stroke)):
           x_count = x_count + 1
           #stroke_buffer = np.vstack((stroke_buffer, stroke[x_num]))
           if x_count % 5 == 0:
               
               plt.plot(stroke[:, 0], stroke[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)
               plt.gca().invert_yaxis();
               plt.axis('off')

               #plt.savefig("tanmay" + '/sketch_' + str(count) + 'points_.jpg', bbox_inches='tight',
                            #pad_inches=0, dpi=1200)
               count = count + 1
               plt.gca().invert_yaxis();


plt.plot(stroke[:, 0], stroke[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)
plt.show()



#print(stroke_list)
#print(a)