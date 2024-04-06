# Take a look at the table below, it is the same data set that we used in the multiple regression chapter, 
# but this time the volume column contains values in liters instead of cm3 (1.0 instead of 1000).

# When your data has different values, and even different measurement units, it can be difficult to compare them. What is kilograms compared to meters? Or altitude compared to time?

# The answer to this problem is scaling. We can scale data into new values that are easier to compare.

# There are different methods for scaling data, in this tutorial we will use a method called standardization.

# The standardization method uses this formula:

# z = (x - u) / s

# Where z is the new value, x is the original value, u is the mean and s is the standard deviation.

# If you take the weight column from the data set above, the first value is 790, and the scaled value will be:
# (790 - 1292.23) / 238.74 = -2.1

# If you take the volume column from the data set above, the first value is 1.0, and the scaled value will be:

# (1.0 - 1.61) / 0.38 = -1.59

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("data2.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X) # scale the weight and volume  
print(scaledX)  # print the result of scaled weight and volume

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

# Predict the CO2 emission from a 1.3 liter car that weighs 2300 kilograms:
predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2) 