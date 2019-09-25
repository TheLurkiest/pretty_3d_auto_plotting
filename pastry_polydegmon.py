import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats

# iris data projected onto 3d scatter plot:

from sklearn.datasets import fetch_openml

fetch_name = ''
print('This program is designed for the user to be able to visualize abstract concepts more easily-- my plugging the values for features into 3d scatter plots to see the ways in which these datasets are structures a little bit more clearly.\n')

fetch_name = input('Enter the name of the dataset you wish to plot onto a 3d scatter plot (a few examples of some choices would be "iris", "wine", "mice" and "boston"):')

tested_out_num = 1
iris = 1

iris = fetch_openml(name=str(fetch_name), version = 'active')


while (tested_out_num <= 10 and type(iris) == int):
    try:
        print('testing... ')
        
        print( 'type is: ' + str(type(iris.data))) 
        if(type(iris) != int):
            print('....done searching')
            break
    except:
        print('no ' +str(tested_out_num))
    tested_out_num += 1
    


print("Here is some info on the dataset you are selecting features from to plot on a 3d scatter plot: " + str(iris.DESCR) + "\n")


#if( .count( str( (str(iris.DESCR))[((str(iris.DESCR)).find(str(elem1 + 1) + '. ')):((str(iris.DESCR)).find('\n    ' + str(elem1 + 2) + '. '))] ) ) )

print("Choose one number at a time, from the following list to select the 3 features you wish to enter as x, y, z coordinate locations for vertices in a 3d scatter plot where individual samples will be placed as vertices.  Please enter the features you prefer in the order of 'x then y then z' one at a time and hit enter between each entry.  Make sure you only enter the NUMBER associated with the feature you want and NOTHING ELSE:")

p_this=''
p_next='\n'

selected_feature_s=''
p_features=[]

if( str(iris.DESCR).count(str('\n    2. ')) < 1 ):
    while(len(p_features) < 3):
        p_this = input('Enter the number of the features you want to use for this 3d model-- although numbers are not listed you can just count up from those shown in description-- in this case starting from 1 for the first feature listed in the description, and increasing incrementally from there: ')
        p_features.append(int(p_this) - 1)
else:
    for elem1 in list(range( iris.data.shape[1] - 1)):
        for elem2 in list(range(iris.data.shape[1])):
            print( '\n' + str(int(elem2) + 1) + '- ' + (str(iris.DESCR))[((str(iris.DESCR)).find(str(int(elem2) + 1) + '. ')):((str(iris.DESCR)).find('\n    ' + str(int(elem2) + 2) + '. '))] )
        str_x = str( (str(iris.DESCR))[((str(iris.DESCR)).find(str(elem1 + 1) + '. ')):((str(iris.DESCR)).find('\n    ' + str(elem1 + 2) + '. '))] )
        selected_feature_s = input('Enter feature number: ')
        p_features.append(int(selected_feature_s) - 1)

print("You have selected the following numbers: " + str(p_features))




iris_points = iris.data[:, p_features]
points = iris_points

hist, binedges = np.histogramdd(points, normed=False)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot(points[:,p_features[0]],points[:,p_features[1]],points[:,p_features[2]],'k.',alpha=0.7)

#Use one less than bin edges to give rough bin location
X, Y = np.meshgrid(binedges[0][1:],binedges[1][:-1])

# these transposes on X and Y fix our weirdly flipped histograms:
X=np.transpose(X)
Y=np.transpose(Y)

#for ct in [0,1,2,3,4,5,6,7,8,9,10,11]: 
for ct in [0,2,4,6,8]: 
	cs = ax1.contourf(X,Y,hist[:,:,ct], zdir='z', offset=binedges[2][ct], level=100, cmap=plt.cm.gnuplot2, alpha=0.5)
# Here are some nice cmaps (more listed here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html): 
# 1. gnuplot2   # 2. RdYlBu_r   # 3. ocean  # 4. viridis    # 5. nipy_spectral  # 6. cividis    # 7. gist_yarg  # 8. inferno/magma

ax1.set_xlim(min(points[:,0]), max(points[:,0]))
ax1.set_ylim(min(points[:,1]), max(points[:,1]))
ax1.set_zlim(min(points[:,2]), max(points[:,2]))











#str_y = str( (str(iris.DESCR))[((str(iris.DESCR)).find('2. ')):((str(iris.DESCR)).find('\n    3. '))] )
#str_z = str( (str(iris.DESCR))[((str(iris.DESCR)).find('3. ')):((str(iris.DESCR)).find('\n    4. '))] )

ax1.set_xlabel('x: sepal length in cm')
ax1.set_ylabel('y: sepal width in cm')
ax1.set_zlabel('z: petal length in cm')
plt.show()

plt.colorbar(cs)

#plt.show()
# ======================================================================================
# ======================================================================================
# ======================================================================================








# ======================================================================================
# perceptron OvR classification of iris dataset:
# ======================================================================================

# start with these features by default: [0, 1, 2]

# multi-class classification via OvR lets us use perceptron for this-- which is what we'll start out testing-- if time allows
# after finishing testing perceptron, we'll try out the "stochastic gradient descent model" provided by scikit-learn after, 
# to see how the two compare and which is better:


# DATA PREP:
# --------

# we have already brought the data in-- as iris.data --earlier in this program

y = iris.target

# MAY need to do this:
#y=np.where(y == 'Iris-setosa', 0, y)
#y=np.where(y == 'Iris-versicolor', 1, y)
#y=np.where(y == 'Iris-virginica', 2, y)

# set X = these 3 features to start:
X = iris.data[:, p_features]

# 1st: split into training and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# 2nd: standardize features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #center around 0 (mean), with a standard deviation of 1.
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 3rd: train our model:
from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=100, eta0=0.1, random_state=42)
ppn.fit(X_train_std, y_train)

# TUNING/RE-TUNING AND EVALUATING OUR MODEL AND THE RESULTS WE GET FROM IT:

# 4th: predict to get Accuracy
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: ' + str((y_test != y_pred).sum()))
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))

# 5th: evaluate model using cross validation:

from sklearn.model_selection import cross_val_score
cross2_acc = cross_val_score(ppn, X_train_std, y_train, cv=4, scoring="accuracy")

print('Using cross-validation it is determined that the average accuracy for features ' + str(p_features[0] +1) + ', ' + str(p_features[1] +1) + ' and '  + str(p_features[2] +1) + ' using the perceptron model is: ' + str( (sum(cross2_acc))/(len(cross2_acc)) ) )

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

print('Precision of perceptron: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average = 'macro'))
print('Recall for perceptron: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average = 'macro'))

print('F1 value calculated for perceptron model: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average = 'macro'))









