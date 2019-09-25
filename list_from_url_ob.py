

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats

# iris data projected onto 3d scatter plot:

from sklearn.datasets import fetch_openml






from seek_data.data_to_3d import RandT1

#import randTurt


ob1 = RandT1()


func1 = ob1.turtlesetup()



print( '\n\nFINAL OUTPUT LIST OF DATASET OPTIONS: ' + '\n\n')
print( '\n\nFINAL OUTPUT LIST OF DATASET OPTIONS: ' + '\n\n')
print( '\n\nFINAL OUTPUT LIST OF DATASET OPTIONS: ' + '\n\n')


print( '\n\nFINAL OUTPUT LIST OF DATASET OPTIONS: ' + str(func1) +'\n\n')







fetch_name = ''
print('This program is designed for the user to be able to visualize abstract concepts more easily-- my plugging the values for features into 3d scatter plots to see the ways in which these datasets are structures a little bit more clearly.\n')

print('If you don\'t yet know the name of the dataset you want to use, or you want to enter an ID directly simply hit enter at the following prompt (and enter a SECOND time at the prompt after that, if you don\'t yet know the ID you want to use either).  After that you will be given the option of entering a search term that will automatically search through the openml website for various datasets that are associated with that particular search term you requested -- as well as their ID-- for and allow you to pick one of them by entering that ID at the prompt that shows up after you make your search.  If the program didn\'t get you any results, or you weren\t satisfied with the results you were given, you can simply hit enter again when an ID is asked for, to make another search-- and can repeat this process again and again until you finally get the results you want.')

fetch_name = input('Enter the name of the dataset you wish to plot onto a 3d scatter plot (a few examples of some choices would be "iris", "wine", or "boston"):')

tested_out_num = 1
iris = 1

iris = fetch_openml(name=str(fetch_name), version = 'active')

id_fetch = ''
fetch_web_search = ''

if(fetch_name == ''):
	id_fetch = input('Enter the ID of the dataset you want, or just hit enter again to search online for a list of dataset ID\'s you can choose from: ')
	if(id_fetch == ''):
		fetch_web_search = input('Enter a new term here for a list of ID\'s of datasets associated with that term you can choose from: ')
		
else:
	print('placeholder for now')










