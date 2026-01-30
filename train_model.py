import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ added (to fix convergence warning properly)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

 # Loading the data set to a pandas dataframe    
credit_card_data = pd.read_csv(r"C:\Users\swaji\Desktop\project1\creditcard.csv")


# First five rows of the data set
print(credit_card_data.head())
print(credit_card_data.tail())

# Dataset information
credit_card_data.info()

# checking the number of missing values in each column
print(credit_card_data.isnull().sum())
# hence we came to know these are no missing values in the data set

# Checking the distribution of legit and fraud transactions
print(credit_card_data['Class'].value_counts())

"""
Now we can see the number of data points for legit and fradulent transactions
284315 Legit transactions and
492 Fraudulent transactions
"""
""" This shows that the dataset is unbalanced
because , out of the two target variables ,the number of legit transactions >> number of fraudulent transactions
If we train a machine learning model with this unbalanced data set , it will be biased towards the legit transactions
and we may end up classifying all transactions as legit and achieve a high accuracy of 99% , but that is not the objective of this project. Hence we need to balance the data set
label 0 represents normal transactions and label 1 represents fraudulent transactions
"""

#Seprerating the data for analysis
legit= credit_card_data[credit_card_data.Class==0]
fraud= credit_card_data[credit_card_data.Class==1]
'''
The entire row with class valur 0 is stored in legit variable and the entire row with class value 1 is stored in fraud variable
'''

print(legit.shape)
print(fraud.shape)

#Now lets get some statistical measures about the data
print(legit.Amount.describe())
print(fraud.Amount.describe())
"""
count is the number of data point of legit and fraud transactions
mean is the average amount for legit transactions
std is the standard deviation
we have the min and max values and also we have the 25%,50% and 75% percentile values. they are in usd
"""
'''
For all fradulant activities the mean amount is quite bigger than the legit transactions mean amount

'''
# Now lets compare the values for both transactions . It compares the mean valur of all the columns.

print(credit_card_data.groupby('Class').mean())
'''
the pattern we can observe is that the values of V1 to V28 columns are different for both the transactions
And this can help us to build a machine learning model to differentiate both the transactions
'''

#Now our next  step is to deal with the unbalanced data set
'''
For this we will use a method called Under-Sampling
It means , we are going to build a sample dataset ,from this original dataset, containing similar distribution of legit and fraudulent transactions
Number of fraudulent transactions = 492
from the legit transactions , out of 284315 datapoints or transactions we will randomly sample 492 transactions,
And we are going to join it with the fradulent transactions
Hence we will have 492 legit transactions and 492 fradulent transactions

Hence the new dataset will be a very good dataset to train our machine learning model
and has a uniform and balanced distribution of both the classes
The benefit is , we can make better predictions using machine learning
'''

#NOw let's take a random sample of our entire dataset
#we are going to use a function called sample() from pandas library
legit_sample= legit.sample(n=492)#IT selects random 492 rows from the legit dataframe which we have already created
#Random sampling is one of the best methods to create a sample dataset from the original dataset
#Now we will concatenate the two DataFrames i.e legit_sample and  and let's create a new dataframe called new_dataset
new_dataset= pd.concat([legit_sample,fraud],axis=0)
# Since we have mentioned the axis is zero , it means we are concatenating the dataframes row-wise and and all the 492 values will be added below the legit_sample dataframe
'''
If we have mentioned axis=1 , it means we are concatenating the dataframes column-wise
But wee need to add this row wise only

'''
#Now let's check the first five rows of the new DataFrame or new dataset
print(new_dataset.head())
# We can also check the last five rows of the new dataset
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())
'''
from this we can conclude that the nature of the dataset has not changed even after sampling. The difference is still there and it
helps our machine learning model to differentiate between the two transactions'''
'''
This step is very important because, it tells us  whether we got a good sample or a bad sample
In case if the mean values were very different from the original dataset , then we would have got a bad sample
'''
# Now we will seperate our dataaset into features and targets(targets are either 0 or 1)
# Once we have seperated it , we can feed it to our machine learning model

#Let us create two variables X and Y and store all the freatures in x and the target variable in Y
x=new_dataset.drop(columns='Class',axis=1)
y=new_dataset['Class']
print(x)#It won't contain the class column
print(y)#It will contain only the class column

#Now we will split the data (features and targets) into training data and testing data
#for this we will use the train_test_split() function which we have imported from sklearn.model_selection library

X_train, X_test, Y_train, Y_test= train_test_split(x,y,test_size=0.2,stratify=y,random_state=2) #thses four variables are arrays
'''
test_size is the amount of testing data we want to split from the original data
i.e 80 percentage of data will be used for training and 20 percentage of data will be used for testing
'''
'''
all the four variables are numpy arrays
All the featues of the training data is stored in X_train
All the features of the testing data is stored in X_test
All the targets of the training data is stored in Y_train
All the targets of the testing data is stored in Y_test
'''
'''
we are using stratify=y to make sure that both training and testing data have similar distribution of legit and fraudulent transactions as in the original dataset'''
print(x.shape,X_train.shape,X_test.shape)

# Now we will train our machine learning model
'''
 we can use different models and check which model gives the best accuracy score
 But generally we use logistic regression model fro binary classification problems'''
 # We have already imported the LogisticRegression model from sklearn library

model= LogisticRegression()#This means we are loading one instance of this logistic regression model into the variable model

# ✅ altered: keep your line, but now we assign it to model properly (previously it was unused)
LogisticRegression(max_iter=1000)
model = LogisticRegression(max_iter=1000)

# ✅ altered: use scaling + logistic regression (best for convergence)
# (we are not deleting your model; we are updating it safely)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

model.fit(X_train,Y_train)

'''
 X_train contains all the features of the training data and Y_train contains the corresponding targets or labels of the training data
'''


'''
What changed (in simple words)

Your LogisticRegression(max_iter=1000) line was not used, so I made model = LogisticRegression(max_iter=1000) (so it actually applies).

Added scaling (StandardScaler) using a Pipeline so the warning won’t show.

Added accuracy prints so you get output.'''
# Traning has been completed and it has took a small time comparitively because the dataset is small after sampling
'''
But for example in deep learning projects where we have millions of data points , it may take hours or even days to train the model
and unlike numericals , images take more time to process and train the model
'''
#But this is a simle project to understand the concepts of machine learning and how to deal with unbalanced datasets
#It is a simple machine learning model to detect credit card fraudulent transactions
#Model evaluation based on accuracy score
#  added: show output so you see results in terminal
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
#Accuracy on training data
print("Training Accuracy:", accuracy_score(Y_train, train_pred))
#Accuracy on testing data
print("Testing Accuracy:", accuracy_score(Y_test, test_pred))
#Multiply the output by 100 to see percentage i.e out of 100 predictions our model can predict that much accurately
'''
we are trying to evaluate the model by providing it the dataset on which it has been trained and checking how accurate it is
and then we are providing the testing data to the model and checking how accurate it is on unseen data
This is done to make sure that our model is not overfitting'''
'''
If there is a big difference between training accuracy and testing accuracy , then our model is overfitting (model is overtrained on the training data) or underfitting(The model is not trained upto the level) the data and it is memorizing the data points
But if both accuracies are nearly equal , then our model is performing well
'''
'''
underfitting :we get very less training data accuracy and high testing data accuracy(The model will be over generalized)
overfitting: we get high training data accuracy and less testing data accuracy(The model will be too specific to the training data and will not generalize well on unseen data)
'''
'''
summary:
import the dependancies(libraries and the functions)
load the csv file into a pandas dataframe
explore the data set(understand the data)
check for missing values
check the distribution of legit and fraudulent transactions
seperate the data into legit and fraud dataframes
get the statistical measures for both the dataframes
compare the values of both the dataframes
create a new balanced dataset by under-sampling the legit dataframe
take a random sample of 492 values from the legit dataframe
concatenate the legit_sample and fraud dataframes row-wise to create a new balanced dataset
seperate the new dataset into features and targets
split the data into training and testing data
train the model
evaluate the model
'''


'''
Since i should upload my project to my github repository but the creditcard.csv file is very large in size(about 150mb) , it is not allowing me to upload the project to github as the allowed limit is 100mb maximum for a single file
so i an deleting the training data from the porject folder as i have already trained my model and got the accuracy
so i will just keep the train_model.py file in the project folder and whenever required i can run this file to train my model again

Hence the new dataset will be a very good dataset to train our machine learning model
and has a uniform and balanced distribution of both the classes
The benefit is , we can make better predictions using machine learning
'''
# ===============================
# Save the trained model + feature columns
# ===============================
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(model, os.path.join(BASE_DIR, "credit_card_fraud_model.pkl"))
joblib.dump(list(X_train.columns), os.path.join(BASE_DIR, "feature_columns.pkl"))

print("\n✅ Saved: credit_card_fraud_model.pkl")
print("✅ Saved: feature_columns.pkl")
print("📁 Saved in:", BASE_DIR)
