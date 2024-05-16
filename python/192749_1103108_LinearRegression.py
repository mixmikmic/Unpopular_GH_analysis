import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from class_vis import prettyPicture, output_image

# Import Data
from ages_net_worths import ageNetWorthData

# Create training and testing dataset (age: X, net_worths: Y)
ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

# Train
def studentReg(ages_train, net_worths_train):
    from sklearn.linear_model import LinearRegression
    
    # create regression
    reg = LinearRegression()
    
    # Train regression
    reg.fit(ages_train, net_worths_train)
    
    return reg
reg = studentReg(ages_train, net_worths_train)

# Visualization
plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")
plt.show()

plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())

# Predictions: Predict X = 27 -> Y
print "net worth prediction", reg.predict([27])

# coefficients: slope(a) and intecept(b) (y = ax + b)
print reg.coef_, reg.intercept_

# stats on test dataset
print "r-squared score: ", reg.score(ages_test, net_worths_test)

# stats on training dataset
print "r-squared score: ", reg.score(ages_train, net_worths_train)

