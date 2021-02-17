from sklearn.datasets import load_iris

iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193]+"\n...")

print("target_names: \n{}".format(iris_dataset['target_names']))

print("feature_names: \n{}".format(iris_dataset['feature_names']))

#Measurements(col) of the flowers(row)
print("Type of data: {}".format(type(iris_dataset['data'])))

print("Shape of data: {}".format(iris_dataset['data'].shape)) #the shape of the data (flowers(sample), features)
print("First five rows of data:\n{}".format(iris_dataset['data'][:5])) #prints the first five rows of data rows are each seperate flower and the col are it's features

print("Type of target: {}".format(type(iris_dataset['target'])))#one dim-array with on entry per flower

print("Shape of target: {}".format(iris_dataset['target'].shape))#prints the shape of the target 

print("Target:\n{}".format(iris_dataset['target']))#The meanings of the numbers are given by the iris['target_names'] array: 0 means setosa, 1 means versicolor, and 2 means virginica.


'''
Definitions:
    -Super Vised Learning Problem: A learning problem that has a set of all ready know measurements (e.t Because we have measurements for which we know the correct species of iris, 
    this is a supervised learning problem)

    -classification problem: a certain set of outcomes (e.t In this problem, we want to predict one of several options (the species of iris).)

    -classes: a specific outcome of classification problem.

    -features: a list of an objects ceratin features(like how tall is this human being) ... Properties of samples

    -samples: individual items (e.g all 150 we have data from)

Gist:
    So far this is our init setup to learning machiene learning. So far we have the dataset iris_dataset that we are going to use to then in return guess the family of an 
    unknown flower

    2/16/2021 *Collin Campbell
    “An Introduction to Machine Learning
    with Python by Andreas C. Müller and Sarah Guido (O’Reilly). Copyright 2017 Sarah
    Guido and Andreas Müller, 978-1-449-36941-5.”

'''