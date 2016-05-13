##################################################
# The Best Kaggle Team
# - Woojin Kim (wk2246)
# - Carlos Espino Garcia (ce2330)
# - Yijing Sun (ys2892)
##################################################
# Description: Stacking prediction model for the Machine Learning 
#  Kaggle competition. Please see the write-up for detailed description 
#  of the procedure. With 2.3 GHz i7 with 8 GB memory, the entire 
#  code takes over 6 hours to run, so please keep in mind before executing.
##################################################

import csv
import numpy as np
import pandas as pd
import random
import sys
from sklearn.model_selection import KFold


if len(sys.argv) < 4:
  print "Usage: python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE"
  exit(1)

DATAFILE = sys.argv[1]
QUIZFILE = sys.argv[2]
OUTPUTFILE = sys.argv[3]

##################################################
##################################################
## Data Processing

#########################
# Load the datasets
train = pd.read_csv(DATAFILE, sep=",")
test_data = pd.read_csv(QUIZFILE, sep=",")
train_data = train.iloc[:,:-1]
train_labels = train.iloc[:,-1]

all_data = pd.concat([train_data, test_data])
train_obs = len(train_data)
test_obs = len(test_data)

# Change dtype of categorical columns
categorical_columns = ['0','5','7','8','9','14','16','17','18','20','23','25','26','56','57','58']
for i in xrange(0,len(categorical_columns)):
    all_data[categorical_columns[i]] = all_data[categorical_columns[i]].astype('category')

#########################
# Only numerical data
print('Processing numerical data...')
all_data_num = all_data.drop(categorical_columns, axis=1)
train_data_num = all_data_num.iloc[0:train_obs,]
test_data_num = all_data_num.iloc[train_obs:,]

train_data_num['div'] = (train_data_num.loc[:,'60'] / train_data_num.loc[:,'59'])
train_data_num.loc[:,'div'] = train_data_num.loc[:,'div'].fillna(0)
test_data_num['div'] = (test_data_num.loc[:,'60'] / test_data_num.loc[:,'59'])
test_data_num.loc[:,'div'] = test_data_num.loc[:,'div'].fillna(0)

#########################
# One-hot encoding for categorical data
## Only categorial data
print('Processing categorical data...')
all_data_cat = pd.get_dummies(all_data[categorical_columns])
train_data_cat = all_data_cat.iloc[0:train_obs,]
test_data_cat = all_data_cat.iloc[train_obs:,]

#########################
## Ignoring two large columns ('slim')
print('Processing categorical data (slim)...')
categorical_columns_slim = ['0','5','7','8','9','14','16','17','18','20','25','26','56','57']

all_data_cat_slim = pd.get_dummies(all_data[categorical_columns_slim])
train_data_cat_slim = all_data_cat_slim.iloc[0:train_obs,]
test_data_cat_slim = all_data_cat_slim.iloc[train_obs:,]

#########################
# Combined sets
print('Combining data...')
train_data_combo = pd.concat([train_data_num, train_data_cat], axis=1)
test_data_combo = pd.concat([test_data_num, test_data_cat], axis=1)

train_data_combo_slim = pd.concat([train_data_num, train_data_cat_slim], axis=1)
test_data_combo_slim = pd.concat([test_data_num, test_data_cat_slim], axis=1)

#########################
# Clear memory
all_data, train, train_data, test_data = None, None, None, None
all_data_num, train_data_num, test_data_num = None, None, None
all_data_cat, train_data_cat, test_data_cat = None, None, None
all_data_cat_slim, train_data_cat_slim, test_data_cat_slim = None, None, None

print('Finished processing!\n')

##################################################
##################################################
## Classifier functions for cross-validation

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.tree import ExtraTreeClassifier

def pred_and_error(model, test_data, test_labels):
    preds = model.predict(test_data)
    error = 1 - sum(preds == test_labels)/float(len(test_labels))
    return preds, error

def cv_run_ada(train_data, train_labels, test_data, test_labels):
    model = AdaBoostClassifier().fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

def cv_run_bag(train_data, train_labels, test_data, test_labels):
    model = BaggingClassifier(max_features=0.394512412319, n_estimators=435,
                              random_state=1, n_jobs=-1).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

def cv_run_et(train_data, train_labels, test_data, test_labels):
    model = ExtraTreesClassifier(n_jobs=-1, min_samples_leaf=2, n_estimators=99,
                                 min_samples_split=3, random_state=1,
                                 max_features=1611, max_depth=None).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

def cv_run_knn(train_data, train_labels, test_data, test_labels, n_neigh):
    model = KNeighborsClassifier(n_neighbors = n_neigh, n_jobs=-1).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

def cv_run_logistic(train_data, train_labels, test_data, test_labels):
    model = LogisticRegression(penalty='l1',
                               C=0.9029677391429398,
                               n_jobs=-1, random_state=1).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

def cv_run_neural(train_data, train_labels, test_data, test_labels):
    model = MLPClassifier(hidden_layer_sizes=900).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

def cv_run_rf(train_data, train_labels, test_data, test_labels):
    model = RandomForestClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=77,
                                   min_samples_split=2, random_state=1, max_features=771,
                                   max_depth=None).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

####################
# Meta-classifiers
def cv_run_bag_meta(train_data, train_labels, test_data, test_labels):
    model = BaggingClassifier(max_features=0.7268891521595635, n_estimators=26,
                              random_state=1, n_jobs=-1).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

def cv_run_et_meta(train_data, train_labels, test_data, test_labels):
    model = ExtraTreesClassifier(max_features=None,
                                 n_jobs=-1, random_state=1).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

def cv_run_rf_meta(train_data, train_labels, test_data, test_labels):
    model = RandomForestClassifier(max_features=0.38988227030541617, min_samples_leaf=4,
                                   min_samples_split=2, n_estimators=112,
                                   random_state=1, n_jobs=-1).fit(train_data, train_labels)
    return pred_and_error(model, test_data, test_labels)

##################################################
##################################################
## Part One - Cross validation predictions and errors

# Split into 5-fold with a set random state for reproducible performance (section 2.1, step 1)
kf = KFold(n_folds=5, shuffle=True, random_state=1)

cv_preds = []
indices = []
for i, (train, test) in enumerate(kf.split(train_data_combo)):
    # Collect the indices used for test sets
    indices = np.concatenate((indices, test))
    
    # Split into train and testing data/labels
    cv_train_data = train_data_combo.iloc[train,:]
    cv_train_data_slim = train_data_combo_slim.iloc[train,:]
    cv_train_labels = train_labels[train]
    
    cv_test_data = train_data_combo.iloc[test,:]
    cv_test_data_slim = train_data_combo_slim.iloc[test,:]
    cv_test_labels = train_labels[test]
    
    # CV predictions & errors for each classifier
    print("Starting fold #{}".format(i+1))
    preds_1, error_1 = cv_run_et(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}".format(error_1))
    preds_2, error_2 = cv_run_rf(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}".format(error_2))
    preds_3, error_3 = cv_run_bag(cv_train_data_slim, cv_train_labels, cv_test_data_slim, cv_test_labels)
    print("Error: {}".format(error_3))
    preds_4, error_4 = cv_run_logistic(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}".format(error_4))
    preds_5, error_5 = cv_run_knn(cv_train_data_slim, cv_train_labels, cv_test_data_slim, cv_test_labels, 1)
    print("Error: {}".format(error_5))
    preds_6, error_6 = cv_run_knn(cv_train_data_slim, cv_train_labels, cv_test_data_slim, cv_test_labels, 2)
    print("Error: {}".format(error_6))
    preds_7, error_7 = cv_run_knn(cv_train_data_slim, cv_train_labels, cv_test_data_slim, cv_test_labels, 4)
    print("Error: {}".format(error_7))
    preds_8, error_8 = cv_run_knn(cv_train_data_slim, cv_train_labels, cv_test_data_slim, cv_test_labels, 8)
    print("Error: {}".format(error_8))
    preds_9, error_9 = cv_run_knn(cv_train_data_slim, cv_train_labels, cv_test_data_slim, cv_test_labels, 16)
    print("Error: {}".format(error_9))
    preds_10, error_10 = cv_run_knn(cv_train_data_slim, cv_train_labels, cv_test_data_slim, cv_test_labels, 32)
    print("Error: {}".format(error_10))
    preds_11, error_11 = cv_run_neural(cv_train_data_slim, cv_train_labels, cv_test_data_slim, cv_test_labels)
    print("Error: {}".format(error_11))
    
    # Collect all the fold predictions together, fold_length * 8 (section 2.1, step 2)
    fold_preds = np.column_stack((preds_1, preds_2, preds_3, preds_4, 
                                  preds_5, preds_6, preds_7, preds_8,
                                  preds_9, preds_10, preds_11))
    
    # Vertically stack the current fold predictions below the previous ones
    if len(cv_preds) == 0:
        cv_preds = fold_preds
    else:
        cv_preds = np.vstack((cv_preds, fold_preds))
        
    print('')

#########################
# Average CV errors for each classifier
cv_labels = train_labels[indices]
for i in xrange(cv_preds.shape[1]):
    print("Method #{}: {}".format(i, 1 - sum(cv_preds[:,i] == cv_labels)/float(len(cv_labels))))
cv_labels = cv_labels.as_matrix()

##################################################
##################################################
## Part Two - Meta-ensemble classification CV

# Convert to Pandas DF for use with sklearn packages (section 2.1, step 3)
cv_preds_stack = pd.DataFrame(cv_preds)

# Split into 10-fold with a set random state for reproducible performance (section 2.1, step 4)
kf = KFold(n_folds=10, shuffle=True, random_state=1)

cv_errors = []
for i, (train, test) in enumerate(kf.split(cv_preds_stack)):
    # Split into train and testing data/labels
    cv_train_data = cv_preds_stack.iloc[train,:]
    cv_train_labels = cv_labels[train]
    cv_test_data = cv_preds_stack.iloc[test,:]
    cv_test_labels = cv_labels[test]
    
    # CV predictions & errors for each classifier
    print("Starting fold #{}".format(i+1))
    preds_0, error_0 = cv_run_ada(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}".format(error_0))
    preds_1, error_1 = cv_run_et_meta(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}".format(error_1))
    preds_2, error_2 = cv_run_rf_meta(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}".format(error_2))
    preds_3, error_3 = cv_run_bag_meta(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}".format(error_3))
    preds_4, error_4 = cv_run_logistic(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}".format(error_4))
    preds_5, error_5 = cv_run_knn(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels, 1)
    print("Error: {}".format(error_5))
    preds_6, error_6 = cv_run_knn(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels, 2)
    print("Error: {}".format(error_6))
    preds_7, error_7 = cv_run_knn(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels, 4)
    print("Error: {}".format(error_7))
    preds_8, error_8 = cv_run_knn(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels, 8)
    print("Error: {}".format(error_8))
    preds_9, error_9 = cv_run_knn(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels, 16)
    print("Error: {}".format(error_9))
    preds_10, error_10 = cv_run_neural(cv_train_data, cv_train_labels, cv_test_data, cv_test_labels)
    print("Error: {}\n".format(error_10))
    
    fold_errors = [error_0, error_1, error_2, error_3, 
                   error_4, error_5, error_6, error_7, 
                   error_8, error_9, error_10]
    cv_errors.append(fold_errors)
    
# Determine best CV error to choose meta-classifier (section 2.1, step 5)
method_errors = pd.DataFrame(cv_errors).mean(axis=0)
for i, method_error in enumerate(method_errors):
    print("Error for method #{}: {}".format(i, method_error))
    
print('\nBest method is #{}: {}'.format(method_errors.idxmin(), method_errors[method_errors.idxmin()]))

##################################################
# Train all models for export (section 2.1, step 6)

print('Model 1')
model = ExtraTreesClassifier(n_jobs=-1, min_samples_leaf=2, n_estimators=99,
                                 min_samples_split=3, random_state=1,
                                 max_features=1611, max_depth=None).fit(train_data_combo, train_labels)
preds_1 = model.predict(test_data_combo)

print('Model 2')
model = RandomForestClassifier(n_jobs=-1, min_samples_leaf=1, n_estimators=77,
                                   min_samples_split=2, random_state=1, max_features=771,
                                   max_depth=None).fit(train_data_combo, train_labels)
preds_2 = model.predict(test_data_combo)

print('Model 3')
model = BaggingClassifier(max_features=0.394512412319, n_estimators=435,
                              random_state=1, n_jobs=-1).fit(train_data_combo_slim, train_labels)
preds_3 = model.predict(test_data_combo_slim)

print('Model 4')
model = LogisticRegression(penalty='l1', C=0.9029677391429398,
                               n_jobs=-1, random_state=1).fit(train_data_combo, train_labels)
preds_4 = model.predict(test_data_combo)

print('Model 5')
model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(train_data_combo_slim, train_labels)
preds_5 = model.predict(test_data_combo_slim)

print('Model 6')
model = KNeighborsClassifier(n_neighbors=2, n_jobs=-1).fit(train_data_combo_slim, train_labels)
preds_6 = model.predict(test_data_combo_slim)

print('Model 7')
model = KNeighborsClassifier(n_neighbors=4, n_jobs=-1).fit(train_data_combo_slim, train_labels)
preds_7 = model.predict(test_data_combo_slim)

print('Model 8')
model = KNeighborsClassifier(n_neighbors=8, n_jobs=-1).fit(train_data_combo_slim, train_labels)
preds_8 = model.predict(test_data_combo_slim)

print('Model 9')
model = KNeighborsClassifier(n_neighbors=16, n_jobs=-1).fit(train_data_combo_slim, train_labels)
preds_9 = model.predict(test_data_combo_slim)

print('Model 10')
model = KNeighborsClassifier(n_neighbors=32, n_jobs=-1).fit(train_data_combo_slim, train_labels)
preds_10 = model.predict(test_data_combo_slim)

print('Model 11')
model = MLPClassifier(hidden_layer_sizes=900).fit(train_data_combo_slim, train_labels)
preds_11 = model.predict(test_data_combo_slim)

# All the base classifier predictions are combined
preds = np.column_stack((preds_1, preds_2, preds_3, preds_4, 
                         preds_5, preds_6, preds_7, preds_8,
                         preds_9, preds_10, preds_11))

# The meta dataset is fed to the best hypertuned meta-classifier (section 2.1, step 7)
model = BaggingClassifier(max_features=0.6297698699152728,
                              n_estimators=60,
                              random_state=1, n_jobs=-1).fit(cv_preds_stack, cv_labels)
results = model.predict(preds)

# Output to file
with open(OUTPUTFILE, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(("Id","Prediction"))
    writer.writerows(zip(range(1,len(results)+1), results))