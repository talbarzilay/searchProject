
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pandas as pd
from heuristics import RFHeuristic, AdaBoostHeuristic, BaggingHeuristic, StackingHeuristic, BaseHeuristic, BootstrappingHeuristic
from preTrainedRegressor import PreTrainedRegressor


isRF =False
isADA=False
isBAG=False
isSTK=False



print('starting....')

# Read data from CSV files
samples_df = pd.read_csv('sample.csv', header=None)
labels_df = pd.read_csv('labels.csv', header=None)

train_features = samples_df.values.tolist()
train_distances = labels_df.values.flatten().tolist()



if isRF:
    print('training rf...')
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(train_features, train_distances)



if isADA:
    print('training xgb...')
    xgb = GradientBoostingRegressor(n_estimators=100)
    xgb.fit(train_features, train_distances)



if isBAG:
    print('training bagging...')
    bag = BaggingRegressor(n_estimators=100, max_samples=0.7)
    bag.fit(train_features, train_distances)




if isSTK:
    print('trained stacking...')

    basic = BaseHeuristic()
    boot = BootstrappingHeuristic()
    boot.load_model()

    # Train Stacking Ensemble with additional heuristics
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=30)),
        ('ada', AdaBoostRegressor(n_estimators=30)),
        ('xgb', GradientBoostingRegressor(n_estimators=30)),
        ('gbr', GradientBoostingRegressor(n_estimators=30)),
        ('bag', BaggingRegressor(n_estimators=30, max_samples=0.7)),
        ('dtr', DecisionTreeRegressor()),
        ('svr', SVR()),
        ('bsc', PreTrainedRegressor(basic)),
        ('bsp', PreTrainedRegressor(boot)),
    ]
    stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    stacking.fit(train_features, train_distances)






print('saving models...')

if isRF:
    rf_heuristic = RFHeuristic(rf)
    rf_heuristic.save_model()

if isADA:
    ada_heuristic = AdaBoostHeuristic(xgb)
    ada_heuristic.save_model()

if isBAG:
    bagging_heuristic = BaggingHeuristic(bag)
    bagging_heuristic.save_model()

if isSTK:
    stacking_heuristic = StackingHeuristic(stacking)
    stacking_heuristic.save_model()

print('done.')


