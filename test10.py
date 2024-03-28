### Multiple Regression
# Multiple regression is like linear regression, but with more than one independent value, meaning that we try to predict a value based on two or more variables.
import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

# make a list of the independent values and call this variable X.
# Put the dependent values in a variable called y
X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2) # We have predicted that a car with 1.3 liter engine, and a weight of 2300 kg, will release approximately 107 grams of CO2 for every kilometer it drives.