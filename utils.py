import math
from ortools.linear_solver import pywraplp
import numpy as np
import matplotlib.pyplot as plt 
import random
import matplotlib.gridspec as gridspec
from bresenham import bresenham


def euclidean_distance(x , y):
    return math.sqrt(sum((a-b)**2 for (a ,b) in zip(x ,y)))

def data_matrix(p1 , p2):
    data_ = []
    for x in p1:
        x_list = []
        for y in p2:
            x_list.append(euclidean_distance(x , y))
        data_.append(x_list)
    return data_



def getAssignment(p1 , p2):
    assert(len(p1) == len(p2)) , "p1 and p2 must have same number of points"
    num_points = len(p1)
    data_ = data_matrix(p1 , p2)
    solver= pywraplp.Solver.CreateSolver("SCIP")

    # Creatinng the optimization variables
    x = {}
    for i in range(num_points):
        for j in range(num_points):
            x[i , j] = solver.IntVar(0 , 1 , '')
    
    # Adding the bijective Constraints
    for i in range(num_points):
        solver.Add(solver.Sum([x[i,j] for j in range(num_points)]) == 1)
    
    for j in range(num_points):
        solver.Add(solver.Sum([x[i,j] for i in range(num_points)]) == 1)
    
    # Objective function 
    objective_terms =[]
    for i in range(num_points):
        for j in range(num_points):
            objective_terms.append(data_[i][j] * x[i , j])
    solver.Minimize(solver.Sum(objective_terms))

    status = solver.Solve()
    ans_dict = {}
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print("Bijection success")
        print("EMD (minimum work) = " , solver.Objective().Value()/num_points , "\n")
        for i in range(num_points):
            for j in range(num_points):
                if x[i , j].solution_value() > 0.5:
                    print(p1[i] , " assigned to " , p2[j])
                    ans_dict[i] = j
        return ans_dict
    else:
        print('No solution found.')



def interpolation(p1 , p2 , lmbda , ans_dict):
    assert 0 <= lmbda <= 1 , "lambda out of bounds"
    n = len(p1)
    intermediate_point_list = []
    for i in range(n):
        j = ans_dict[i]
        point1 = p1[i]
        point2 = p2[j]
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        intermediate_point = (1 - lmbda) * point1 + lmbda * point2
        print("\n")
        print("Point1 is:" , point1)
        print("Point2 is:" , point2)
        print("Interpolated point : " , intermediate_point)
        print("-" * 20)
        intermediate_point_list.append(intermediate_point.tolist())
    return intermediate_point_list


def get_interpolation(p1 , p2):
    print("Starting.....\n")
    ans_dict = getAssignment(p1 , p2)
    visualize_numpy_points(p1 , "cake")
    visualize_numpy_points(p2 , "tree")
    if ans_dict:
        for lmbda in [0.1 , 0.2, 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9]:
            intermediate_point_list = interpolation(p1 , p2 , lmbda ,ans_dict)
            visualize_numpy_points(intermediate_point_list , lmbda)
        
    else:
        print("Interpolation not possible")

def get_interpolation_with_plots(p1 , p2 , title , save_path):
    print("Starting.....\n")
    ans_dict = getAssignment(p1 , p2)
    fig = plt.figure()
    fig.tight_layout() 
    gs = gridspec.GridSpec(2, 6)
    if ans_dict:
        counter = 0
        for lmbda in [0.0 , 0.1 , 0.2, 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.0]:
            intermediate_point_list = interpolation(p1 , p2 , lmbda ,ans_dict)
            data_x = [coordinate[0] for coordinate in intermediate_point_list]
            data_y = [coordinate[1] for coordinate in intermediate_point_list]

            if counter <= 4:
                ax = plt.subplot(gs[0 , counter%5])
                ax.set_facecolor("black")
                ax.scatter(data_x , data_y , s=10 , color="white")
                ax.set_title(lmbda)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            elif counter <=9 :
                ax = plt.subplot(gs[1 , counter%5])
                ax.set_facecolor("black")
                ax.scatter(data_x , data_y  , s=10 , color="white")
                ax.set_title(lmbda)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            else:
                ax = plt.subplot(gs[1 , 5])
                ax.set_facecolor("black")
                ax.scatter(data_x , data_y , s=10 , color="white")
                ax.set_title(lmbda)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            fig.add_subplot(ax)
            counter +=1 
        fig.suptitle(title)
        plt.subplots_adjust(top = 0.9, bottom = 0.1, right = 0.9, left = 0.125, hspace = 0.2, wspace = 0.2)
        plt.savefig(save_path)

    else:
        print("Interpolation not possible")



def visualize_numpy_points(data , title):
    data_x = [coordinate[0] for coordinate in data ]
    data_y = [coordinate[1] for coordinate in data]
    axs = plt.axes()
    axs.set_facecolor("black")
    plt.scatter(data_x , data_y , color="white" , s=10)
    plt.title(title)
    plt.show()



def visualize_numpy_points_two(p):
    stroke_buffer = p[0]
    for x_num in stroke_buffer:
                plt.plot(p[:, 0], p[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)
                plt.gca().invert_yaxis();
                plt.axis('off')

                plt.savefig( "tanmay" + '/sketch_' + 'points_.jpg', bbox_inches='tight',
                            pad_inches=0, dpi=1200)
                plt.gca().invert_yaxis();

def open_numpy_file(path):
    data_ = np.load(path)
    # Removing pen movements
    data_ = data_[: , :2]
    return data_




def strategy1(p1 , p2):
    random.seed(5)

    num_point_1 = p1.shape[0]
    num_point_2 = p2.shape[0]

    if num_point_1 > num_point_2:
        point_1_list = []
        index_list = list(range(num_point_1))
        sampled_list = random.sample(index_list , num_point_2)
        for index in sampled_list:
            point_1_list.append(p1[index])
        point_2_list = p2.tolist()
    elif num_point_2 > num_point_1:
        point_2_list = []
        index_list = list(range(num_point_2))
        sampled_list = random.sample(index_list , num_point_1)
        for index in sampled_list:
            point_2_list.append(p2[index])
        point_1_list = p1.tolist()
    else:
        point_1_list = p1.tolist()
        point_2_list = p2.tolist()
    
    return point_1_list , point_2_list


def strategy2(p1 , p2):
    random.seed(0)

    num_point_1 = p1.shape[0]
    num_point_2 = p2.shape[0]
    p1 = p1.tolist()
    p2 = p2.tolist()

    if num_point_1 > num_point_2:
        difference = num_point_1 - num_point_2
        index_list = random.choices(list(range(num_point_2)) , k=difference)
        for index in index_list:
            point = p2[index]
            p2.append(point)

    elif num_point_2 > num_point_1:
        difference = num_point_2 - num_point_1        
        index_list = random.choices(list(range(num_point_1)), k=difference)
        for index in index_list:
            point = p1[index]
            p1.append(point)

    return p1 , p2
def mydrawPNG(vector_image):
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    final_list = []
    point_set = 0 
    for i in range( 0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
                point_set += 1

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        for ele in range(len(cordList)):
            cordList[ele] = list(cordList[ele])
            cordList[ele].append(point_set)
        final_list.extend([list(i) for i in cordList])
        
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
    return final_list

def preprocess(sketch_points, side=800):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([side, side])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def strategy3(p1 , p2 , max_points):
    p1 = preprocess(p1 , 256)
    p2 = preprocess(p2 , 256)
    p1 = mydrawPNG(p1)
    p2 = mydrawPNG(p2)
    assert len(p1) > max_points , "Max points exceed lenght of p1 := {}".format(len(p1))
    assert len(p2) > max_points , "Max points exceed lenght of p2 := {}".format(len(p2))

    point_1_list = []
    index_list = list(range(len(p1)))
    sampled_list = random.sample(index_list , max_points)
    for index in sampled_list:
        point_1_list.append(p1[index])

    point_2_list = []
    index_list = list(range(len(p2)))
    sampled_list = random.sample(index_list , max_points)
    for index in sampled_list:
        point_2_list.append(p2[index])
    
    return point_1_list , point_2_list


def strategy4(p1  ,p2):
    num_point_1 = p1.shape[0]
    num_point_2 = p2.shape[0]
    if num_point_1 > num_point_2:
        p2 = preprocess(p2 , 256)
        p2 = mydrawPNG(p2)
        point_2_list = []
        index_list = list(range(len(p2)))
        sampled_list = random.sample(index_list , num_point_1)
        for index in sampled_list:
            point_2_list.append(p2[index])
        p1 = p1[: , :2].tolist()
        return p1 , point_2_list

    if num_point_1 < num_point_2:
        p1 = preprocess(p1 , 256)
        p1 = mydrawPNG(p1)
        point_1_list = []
        index_list = list(range(len(p1)))
        sampled_list = random.sample(index_list , num_point_2)
        for index in sampled_list:
            point_1_list.append(p1[index])
        p2 = p2[: , :2].tolist()
        return point_1_list , p2
    else:
        return p1[: , :2].tolist() , p2[: , :2].tolist()




    #Point set 1
p1 = [
      (0, 0),
       (0, 1),
        (1, 0),
        (1, 1),
    ]

    #Point set 2
p2 = [
        (0.1, 0.1),
        (0.1, 1.1),
       (1.1, 0.1),
       (1.1, 1.1),
   ]
#import torch 
#p1 = torch.randn(500 , 2).tolist()
#p2 = torch.randn(500 ,2).tolist()
#get_interpolation(p1 , p2 , 0.6)
