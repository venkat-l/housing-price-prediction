# Regression Project: Predicting the House Prices from a dataset here http://lib.stat.cmu.edu/datasets/boston

# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)

# Descriptive and Visual statistics to get more information on the data and identify any possible relations between columns.

# Descriptive statistics

# shape
print(dataset.shape)
# types
print(dataset.dtypes)
# head
print(dataset.head(20))
# descriptions, change precision to 2 places
set_option('precision', 1)
print(dataset.describe())
# We now have a better feeling for how different the attributes are. The min and max values as well as the means vary a lot. 
# We are likely going to get better results by rescaling the data in some way.
# Now let us find the correlation between the columns of the data
set_option('precision', 2)
print(dataset.corr(method='pearson'))
# This is interesting. We can see that many of the attributes have a strong correlation (e.g.> 0:70 or < 0:70).
# NOX and INDUS with 0.77.
# DIS and INDUS with -0.71.
# TAX and INDUS with 0.72.
# AGE and NOX with 0.73.
# DIS and NOX with -0.78.
# It also looks like LSTAT has a good negative correlation with the output variable MEDV with a value of -0.74.


# Data visualizations

# histograms
dataset.hist()
pyplot.show()
# density
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
pyplot.show()
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# There is a lot of structure in this dataset. We need to think about transforms that we could use later to better expose 
#the structure which in turn may improve modeling accuracy.

# Prepare Data

# Split-out validation dataset. This is a sample of the data that we hold back from our analysis and modeling. 
# We use it right at the end of our project to confirm the accuracy of our final model.
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Evaluate Algorithms - Baseline
# We have no idea what algorithms will do well on this problem. Let's design our test harness. We will use
# 10-fold cross-validation. The dataset is not too small and this is a good standard test harness
# configuration. We will evaluate algorithms using the Mean Squared Error (MSE) metric. MSE
# will give a gross idea of how wrong all predictions are (0 is perfect).
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# Let's create a baseline of performance on this problem and spot-check a number of different
# algorithms. We will select a suite of different algorithms capable of working on this regression
# problem. The six algorithms selected include:
# Linear Algorithms: Linear Regression (LR), Lasso Regression (LASSO) and ElasticNet (EN).
# Nonlinear Algorithms: Classification and Regression Trees (CART), Support Vector Regression (SVR) and k-Nearest Neighbors (KNN).

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# Let us evaluate each model in turn. The algorithms will all use default tuning parameters. Let's compare the algorithms. 
# We will display the mean and standard deviation of MSE for each algorithm as we calculate it and  collect the results 
# for use later.
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# It looks like LR has the lowest MSE, followed closely by CART. We can see similar distributions for the regression 
# algorithms and perhaps a tighter distribution of scores for CART.

# The differing scales of the data is probably hurting the skill of all of the algorithms and perhaps more so for SVR and KNN.
# So let us run the same Algorithms on a Standardized dataset. To avoid leakage we use pipelines 
# that standardize the data and build the model for each fold in the cross-validation test harness.
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Running the example provides a list of mean squared errors. We can see that scaling did have an effect on KNN, 
# driving the error lower than the other models.
# We can also see that KNN has both a tight distribution of error and has the lowest mean error score.

# Now let us tune the Algo with different values of K to see if it can do better? The default value for the
# number of neighbors in KNN is 7. We can use a grid search to try a set of different numbers of neighbors and see if
# we can improve the score. The below example tries odd k values from 1 to 21, an arbitrary
# range covering a known good value of 7. Each k value (n neighbors) is evaluated using 10-fold
# cross-validation on a standardized copy of the training dataset.
# KNN Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# You can see that the best for k (n neighbors) is 3 providing a mean squared error of -18.172137, the best so far.

# Now let us see, if this can be further tuned with Ensembles
# In this section we will evaluate four different ensemble machine learning algorithms, two boosting and two bagging methods:
# Boosting Methods: AdaBoost (AB) and Gradient Boosting (GBM).
# Bagging Methods: Random Forests (RF) and Extra Trees (ET).
# We will use the same test harness as before, 10-fold cross-validation and pipelines that standardize the training data for each fold.
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Running the example calculates the mean squared error for each method using the default parameters. 
# We can see that we're generally getting better scores than our linear and nonlinear algorithms in previous sections.
# It also looks like Gradient Boosting has a better mean score.

# Now let us look at tuning the scaled GBM with different n_estimators. 
# Each setting is evaluated using 10-fold cross-validation.
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Based on the output above, we can find that the best configuration was 
# n_estimators=400 a mean squared error of -9.356471, about 0.65 units better than the untuned method.

# Finalize Model and Make predictions on validation dataset
# Now that we finalized our dataset (scaled), our Algorithm (GBM) and the parameter (n_estimators=400), let us use it to
# make the predictions
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)
# transform the validation dataset to scale and predict
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))
# We can see that the estimated mean squared error is 11.8, close to our estimate of -9.3