import heapq
import time
import random
from collections import deque
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pandas as pd
from heuristics import RFHeuristic, AdaBoostHeuristic, StackingHeuristic


print('starting....')

# Read data from CSV files
samples_df = pd.read_csv('sample.csv', header=None)
labels_df = pd.read_csv('labels.csv', header=None)

print('loaded sample....')

train_features = samples_df.values.tolist()
train_distances = labels_df.values.flatten().tolist()

print('pre proccesed sample....')

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(train_features, train_distances)

print('trained rf...')

# Train AdaBoost
ada = AdaBoostRegressor(n_estimators=100)
ada.fit(train_features, train_distances)

print('trained ada...')

# Train Stacking Ensemble with additional heuristics
estimators = [
    ('rf', RandomForestRegressor(n_estimators=30)),
    ('ada', AdaBoostRegressor(n_estimators=30)),
    ('gbr', GradientBoostingRegressor(n_estimators=30)),
    ('dtr', DecisionTreeRegressor()),
    ('svr', SVR())
]
stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking.fit(train_features, train_distances)

print('trained stacking...')




# models

rf_heuristic = RFHeuristic(rf)
stacking_heuristic = StackingHeuristic(stacking)
ada_heuristic = AdaBoostHeuristic(ada)

#save models

rf_heuristic.save_model()
stacking_heuristic.save_model()
ada_heuristic.save_model()

print('done.')


