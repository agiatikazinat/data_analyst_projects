import pandas as pd 
import numpy as np                     
import matplotlib.pyplot as plt   
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression

# MODULE 1: IMPORTING DATA SETS ==============================================================================================================================================================================================================================

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(filepath)

# display the data types of each column using the function dtypes
print("display the data types of each column ", df.info())
print("statistical summary of the dataframe: ", df.describe())


# MODULE 2: DATA WRANGLING ==============================================================================================================================================================================================================================

# drop the columns 'id' and 'Unnamed: 0' 
df.drop('id', axis = 1, inplace = True)
df.drop('Unnamed: 0', axis = 1, inplace = True)
print("after dropping the columns: ", df.describe())

print("number of NaN values for the column bedrooms: ", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms: ", df['bathrooms'].isnull().sum())

mean = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean, inplace=True)

mean = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean, inplace=True)

print("The number of NaN values for the column bedrooms: ", df['bedrooms'].isnull().sum())
print("The number of NaN values for the column bathrooms: ", df['bathrooms'].isnull().sum())

# MODULE 3: EXPLORATORY DATA ANALYSIS ==============================================================================================================================================================================================================================

# use the method value_counts() to count the number of houses with unique floot values,
# use the method to_frame() to convert it to a data frame 
print(df['floors'].value_counts().to_frame())

sns.boxplot(x = df['waterfront'], y = df['price'])
plt.title("Waterfront view affect price or not")

sns.regplot(x = df['sqft_above'], y = df['price'])
plt.title("Feature sqft_above positively correlated with price")
plt.show()

# feature other than price that is most correlated with price
print(df.corr()['price'].sort_values())

# MODULE 4: MODEL DEVELOPMENT ==============================================================================================================================================================================================================================

x = df[['long']]
y = df['price']
lm = LinearRegression()
lm.fit(x, y)
print("r2 score of the model: ", lm.score(x, y))

# fit a linear regression model to predict the price using the feature sqft_living 
# calcualte the r2
x = df[['sqft-living']]
y = df['price']
lm.fit(x, y)
print('R2 score of the model use sqft_living to predict price: ', lm.score(x, y))

# Fit a linear regression model to predict price using the list of features
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     
z = df[features]
y = df['price']
lm.fit(z, y)
print("R2 score of the model predicting price using the rest of other features: ", lm.score(z, y))

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
from sklearn.metrics import r2_score, mean_absolute_error
pipeline = Pipeline(Input)
z = z.astype('float')
pipeline.fit(z, y)
yhat = pipeline.predict(z)
print("r2 score of the pipeline: ", r2_score(yhat, y))

# MODULE 5: MODEL EVALUATION AND REFINEMENT ==============================================================================================================================================================================================================================
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

from sklearn.linear_model import Ridge

RR = Ridge(alpha = 0.1)
RR.fit(x_train, y_train)
yhat = RR.predict(x_test)
print("R2 score of the ridge regression using the training data: ", r2_score(yhat, y_test))

pr = PolynomialFeatures(degree = 2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RR.fit(x_train_pr, y_train)
yhat= RR.predict(x_test_pr)
print("R2 score of the training data after perform a second order polynomial transform on the training and testing data: ", r2_score(yhat, y_test))
