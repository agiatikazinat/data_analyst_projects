import numpy as np             
import pandas as pd  
import seaborn as sns 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler 
import matplotlib.pyplot as plt   
from tqdm import tqdm 
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline 

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'

df = pd.read_csv(filepath)

# TASK 1: IMPORT THE DATASET =====================================================================================================================================================================================

df.columns = ['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region', 'charges']
df.replace('?', np.nan, inplace=True)

# TASK 2: DATA WRANGLING =====================================================================================================================================================================================

print("information of the data: ", df.info())

# handle missing data for age and smoker 
# For continuous attributes (e.g., age), replace missing values with the mean.
# For categorical attributes (e.g., smoker), replace missing values with the most frequent value.
# Update the data types of the respective columns.
is_smoker = df['smoker'].value_counts().idxmax()
df['smoker'].replace(np.nan, is_smoker, inplace=True)

mean_age = df['age'].astype('float').mean(axis = 0)
df['age'].replace(np.nan, mean_age, inplace=True)

df[['smoker', 'age']] = df[['smoker', 'age']].astype('int')

print("after replace missing data: ", df.info())

# Round up the decimal value of charges to 2 
df['charges'] = np.round(df['charges'], 2)
print("after round up: ", df.head())

# TASK 3: EXPLORATORY DATA ANALYSIS (EDA) =====================================================================================================================================================================================

# show the regression plot for charges with respect to bmi 
sns.regplot(x = 'bmi', y = 'charges', data = df)
plt.ylim(0, )
# show the box plot for charges with respect to smoker
sns.boxplot(x = 'smoker', y = 'charges', data = df)
plt.show()

# show the correlation of the dataset
print("Correlation of the dataset: ", df.corr())

# TASK 4: MODEL DEVELOPMENT =====================================================================================================================================================================================

# fit a linear regression model that use to predict the charges value, just by uisng the smoker attribute of the data
# print the R^2 score

lr = LinearRegression()
x = df[['smoker']]
y = df['charges']
lr.fit(x, y)
print("r2 score of the model smoker and charges: ", lr.score(x, y))

# fit a linear regression model that use to predcit charges value just by using all other attributes of the data
# print R^2 score of the model 
lr = LinearRegression()
x = df[['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region']] 
y = df['charges']
lr.fit(x, y)
print("R2 score of the new model using all other atributes to predict charges: ", lr.score(x, y))

# create a training pipeline that uses StandardScaler(), PolynomialFeatures() and LinearRegression() to create a model that can predict the charges
# value using all other attributes

input = [('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('linear', LinearRegression())]
pipeline = Pipeline(input)
x = x.astype('float')
pipeline.fit(x, y)
yhat = pipeline.predict(x)
print("R2 score of the pipeline: ", r2_score(yhat, y))

# TASK 5: MODEL REFINEMENT =====================================================================================================================================================================================

# split the data into training an testing subsets, assuming that 20% of the dasta will be reserved for testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# initialize a ridge regressor that use hyperparameter alpha = 0.1
# fit the model using the training data subset
# print the R2 score for the testing data 
RR = Ridge(alpha = 0.1)
RR.fit(x_train, y_train)
yhat = RR.predict(x_test)
print("R2 score fo the ridge regression model: ", r2_score(yhat, y_test))

# apply the polynomial transofrmation to the training parametes with degree = 2
# use transformed feature set to fit the same regression model using the training subset
# print the R2 score for the testing data 
pr = PolynomialFeatures(degree = 2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RR.fit(x_train_pr, y_train)
yhat = RR.predict(x_test_pr)
print("After applying the polynomial features, the r2 score of the model is ", r2_score(yhat, y_test))