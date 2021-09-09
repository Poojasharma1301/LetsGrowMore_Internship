# LetsGrowMore_Internship
## TASK 1 : Iris Flower Prediction Machine Learning Model
### Dataset Information
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

### Attribute Information:-
-sepal length in cm<br>
-sepal width in cm<br>
-petal length in cm<br>
-petal width in cm<br>
### species: 
1.Iris Setosa<br>
2.Iris Versicolour <br>
3.Iris Virginica<br>
## Steps follow for solving this Machine learning Problem:-
1. Import Module 
2. Loading the dataset
3. Preprocessing the dataset 
4. Exploratory data analysis 
5. Correlation Matrix
6. Label encoder
7. Model traning 
8. Apply different Models and select Perfect one
## Libraries Used
1.pandas<br>
2.matplotlib<br>
3.seaborn<br>
4.scikit-learn<br>
## Algorithms Used
1.Logistic Regression<br>
2.K-Nearest Neighbors<br>
3.Decision Tree<br>
## 🏆 Best Model Accuracy: 95.50%<br>

## TASK 2 : Stock Market Prediction and Forecasting Using LSTM
### Dataset Information
Data given total trade quantity and turnover and we have to forecaste for next days what is the turnover in lacs.Features is given as Date , open , high , low , last and close . 
LSTM - Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. This is a behavior required in complex problem domains like machine translation, speech recognition, and more. LSTMs are a complex area of deep learning.

### Steps Follow for solving this Problem 
1. We will collect the stock data
2. Preprocess the Data - Train and Test
3. Create an Stacked LSTM Model 
4. Predict the test data and plot the output
5. Predict the future 30 days and plot the output 
6. Cross Validation Using Randomseed
7. Data Preprocessing 
8. Analysis for next 30 days 

## TASK 3 : Music Recommendation System Machine Learning Model
Music recommender system can suggest songs to users based on their listening pattern.

### Dataset Information
In the dataset there are five different files is given for visualization and find meaningful insights from these . They are train dataset , test dataset , members dataset , songs dataset and songs extra information dataset . 

## Library Used :- 
1. Pandas
2. NumPy
3. Matplotlib 
4. IPython
5. Seaborn 
6. Warnings
7. Scikit learn

## Steps follow for solving this problem :-
1. Importing Necessary libraries 
2. Importing all the dataset 
3. Complete Analysis of all the dataset files 
4. Exploratory Data Analysis 
5. Handle the missing values
6. Merge all the Dataset
7. Label Encoding on Categorical Data
8. Split the data into train and test Data
9. Model Building and Selection of best Model

Model Used:-
1. Logistics Regression
2. Random forest Classifiers 
3. CLF
4. Decision Tree Classifier 
5. KNeighborsClassifier

## 🏆Best Accuracy score with Random forest Classifier

## TASK 4 : Image to Pencil Sketch with Python

We need to read the image in RBG format and then convert it to a grayscale image. This will turn an image into a classic black and white photo. Then the next thing to do is invert the grayscale image also called negative image, this will be our inverted grayscale image. Inversion can be used to enhance details. Then we can finally create the pencil sketch by mixing the grayscale image with the inverted blurry image. This can be done by dividing the grayscale image by the inverted blurry image. Since images are just arrays, we can easily do this programmatically using the divide function from the cv2 library in Python.

## Steps follow for solving this problem :- 
1. Import the image
2. Read the image in RBG format
3. Convert it into grayscale image
4. Invert the image called negative image 
5. Finally create pencil sketch by mixing grayscale and negative image

## TASK 5 : Exploratory Data Analysis on Dataset - Terrorism
### Problem Statement :-
Perform ‘Exploratory Data Analysis’ on dataset ‘Global Terrorism’.What all security issues and insights you can derive by EDA?.As a security/defense analyst, try to find out the hot zone of terrorism.

## Dataset: https://bit.ly/2TK5Xn5

## Steps follow for solving this problem:-
1.Import Necessary Libaries<br>
2.Import the Dataset<br>
3.Filtering Countries with most terrorist attacks.<br>
4.Filtering country names from countries with most terrorist attacks<br>
5.Plotting the data to the bar graph for data of countries with most terrorist attacks.<br>
6.Analysis Using Pie charts<br>

## Library Used :- 
1. Pandas
2. NumPy
3. Matplotlib 
4. IPython
5. Seaborn 
6. Warnings
7. Scikit learn

## TASK 6 : Prediction using Decision Tree  Algorithm
## Dataset Information:-
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.Same Iris folwer dataset which is called the "Hello World" of machine learning is given.
## Problem Statement:-
Create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to  predict the right class accordingly.
### Attribute Information:-
-sepal length in cm<br>
-sepal width in cm<br>
-petal length in cm<br>
-petal width in cm<br>
### species: 
1.Iris Setosa<br>
2.Iris Versicolour <br>
3.Iris Virginica<br>
## Steps follow for solving this Problem :-
1. Importing all necessary libraries<br>
2. Importing Dataset<br>
3. Visualization<br>
4. Applying Decision tree algorithm<br>
5. Making Desicion tree view using plot_tree <br>
## Libraries Used
1.pandas<br>
2.matplotlib<br>
3.seaborn<br>
4.scikit-learn<br>
5.Warnings
## Observation:-
1 .Sepal Length and Sepal Width are Normally Distributed.<br>
2 .Petal Length and Petal Width both are rightly Skewed.<br>
## Model Used:-
1. Decision Tree Algorithm<br>
## 🏆 Best Model Accuracy: 97.77%<br>

## TASK 7 : Develop A Neural Network That Can Read Handwriting
we will classify handwritten digits using a simple neural network which has only input and output layers. We will than add a hidden layer and see how the performance of the model improves
### Problem Statement:-
Begin your neural network machine learning project with the MNIST Handwritten Digit Classification Challenge and using Tensorflow and CNN. It has a very user-friendly interface that’s ideal for beginners.
### Dataset Information
### MNIST Datasets
MNIST stands for “Modified National Institute of Standards and Technology”. It is a dataset of 70,000 handwritten images. Each image is of 28x28 pixels i.e. about 784 features. Each feature represents only one pixel’s intensity i.e. from 0(white) to 255(black). This database is further divided into 60,000 training and 10,000 testing images.
### Libraries Used:-
1.Numpy<br>
2.Matplotlib<br>
3. Tensorflow<br>
4. Keras<br>
### Model Used:-
Keras Sequential model<br>
### Steps follow for solving this problem:-
1.Import the Necessary libraries <br>
2. Import the dataset<br>
3. Create a Model<br>
4. Pre-process the data<br>
5. Compile the Model <br>
6. Train the Model <br>
7. Evaluate the Model<br>

### Dataset Link: https://en.wikipedia.org/wiki/MNIST_database
## 🏆 Best Model Accuracy: 98.32%<br>

## TASK 8 : Next Word Prediction
Most of the keyboards in smartphones give next word prediction features; google also uses next word prediction based on our browsing history. So a preloaded data is also stored in the keyboard function of our smartphones to predict the next word correctly.
### Problem Statement:-
Using Tensorflow and Keras library train a RNN, to predict the next word. 
### Libraries Used:-
1.Numpy<br>
2.Matplotlib<br>
3.Pickle<br>
4. Keras<br>
### Model Used:-
LSTM(Recurrent Neural networks) 
### Steps follow for solving this proble:-
1. Import the Necessary libraries<br>
2. Import the dataset<br>
3. Feature engineering<br>
4. Building the Recurrent Neural Network<br>
5. Training the Model <br>
6. Evaluate the Model<br>
7. Testing the model<br>

### Dataset Link: https://drive.google.com/file/d/1GeUzNVqiixXHnTl8oNiQ2W3CynX_lsu2/view



