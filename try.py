from itertools import count
import pickle

file = open("ShoeV2_Coordinate" , 'rb')
object_file = pickle.load(file)
file.close()

a = list(object_file.keys())
counter = 0
for sketch in range(len(a)):
    lngth = len(object_file[a[sketch]])
    counter +=1 

print(counter)


        






#print(list(object_file.keys())[12084])
#print(len(object_file[list(object_file.keys())[8206]]))

#print(list(object_file.keys())[3158])
#print(len(object_file[list(object_file.keys())[10080]]))
