# The Best Kaggle Team
[Woojin Kim](http://woojink.com), [Carlos Espino Garcia](http://www.cespinog.com/), [Yijing Sun](https://github.com/Phoebe0711)

This project was completed for the in-class Kaggle competition in COMS 4721: Machine Learning for Data Science at Columbia University, Spring 2016. We finished [second place](https://inclass.kaggle.com/c/coms4721spring2016/leaderboard) with the approach described below.

## Data preprocessing and feature design
The given dataset consisted of both numerical and categorical predictors. In order to use these variables effectively in `scikit-learn`, we split the dataset into two for preprocessing.

To process the categorical variables, we combined the training and the testing data to observe all the possible categories at once, then we used the `get_dummies()` function in the `pandas` package to perform the one-hot encoding to convert the categorical variables into numerical zero-one states. Because two of the predictors contributed a total of 5,121 categories resulting in as many predictors for the processed dataset, we also created an alternate categorical section for use with predictor models that struggled with handling as many predictor columns (both concerning running time and memory use). The numerical portion was left untouched and was appended to the categorical portion after one-hot encoding.

| Dataset | Description |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| numerical + categorical      | Numerical features with all categorical variables                                                                                         |
| numerical + categorical slim | Numerical features with all categorical variables excluding columns 23 and 58 which have 3,031,and 2,090 categories/columns, respectively |

Many variations were considered in terms of feature engineering, including square root, logarithms, second-order interactions, as well as divisions, additions, and subtractions. Most of the methods did not contribute to lower cross-validation error rates; however, one method—dividing column 60 by column 59—improved the predictions slightly, so this engineered feature was kept for the final prediction.

## Model Selection
### Stacking
For the overall prediction, we used ensembling through stacking. Known classifiers (e.g. random forest, logistic regression, *k*NN) were used as base classifiers and the results obtained through these methods were fed to another meta-classifier that output our final predictions.

Although these advanced base classifiers may be efficient at finding the correct predictions for the most part, we found that an additional layer of meta-ensembling provided different 'perspectives' on the dataset that individual methods cannot provide that result in slight increase in prediction performance (e.g. distance information that *k*NN provides that logistic regression would not take into account). Below we describe how the base classifiers were selected and the hyperparameters were tuned.

Stacking itself was performed as follows:

* The training set is split into a standard 5-fold split
* All the chosen base classifiers evaluate and output the predictions for identical splits
* The predictions from all the classifiers and folds are combined into a new meta-dataset
* The meta-dataset is input into a meta 10 folds, evaluated with a similar set of classifiers as step 2
* The top performing meta-classifier is tuned further to determine the final meta-classifier
* The same chosen base classifiers from step 2 evaluate the test set
* The output of the test set predictions are input into the meta-classifier from step 5 for the final predictions

### Hyperparameter Tuning
We used a combination of `GridSearchCV()` and `RandomizedSearchCV()` to explore the possible hyperparameter space for obtaining the best results for each model. Although both of these methods can be used to explore similar range of parameters, we found that the random searching nature of `RandomizedSearchCV()` could reach better set of results in a shorter period of time than the exhaustive method `GridSearchCV()` takes.

The number of iterations taken and the range of parameters varied with the model being optimized, but in general we used 3-fold cross-validation with 30 iterations to determine the best set of hyperparameters. Both the hyperparameters for the base classifiers and the meta-classifiers were obtained this way.

### Predictor evaluation
Each of the classifiers was evaluated through 5-fold cross-validation, split with `StratifiedKFold()` for better distribution of classes in each fold. The best set of hyperparameters obtained for each model were evaluated with a standardized 5-fold split to have a consistent comparison between the models and to obtain an idea of performance entering the meta-ensemble stacking.

Stacking itself was evaluated with a similar method. The combined set of predictions from all the chosen models were put into a 10-fold cross-validation step, evaluated with nearly identical set of models. Here, the most promising of these methods (typically bagging and random forest classifiers) were tuned further for the best set of hyperparameters. The method with the best cross-validation error at this point was used for the final meta-classifier.

### Classifier selection
We tried the following existing algorithms available in the `scikit-learn` package:

| Algorithm | Function |
|---|---|
| Adaboost | `sklearn.ensemble.AdaBoostClassifier` |
| Bagging | `sklearn.ensemble.BaggingClassifier` |
| Extra Trees | `sklearn.ensemble.ExtraTreesClassifier` |
| Random Forests | `sklearn.ensemble.RandomForestClassifier ` |
| Gradient Boosting | `sklearn.ensemble.GradientBoostingClassifier` |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` |
| Perceptron | `sklearn.linear_model.SGDClassifier` |
| Naive Bayes | `sklearn.naive_bayes.MultinomialNB ` |
| *k*NN | `sklearn.neighbors.KNeighborsClassifier ` |
| Neural Networks | `sklearn.neural_network.MLPClassifier ` |
| SVM | `sklearn.svm.SVR ` |

Initially, we tried training all the above algorithms with either the default parameters or with minimal hyperparameter tuning and moved forward loosely based on (a) cross-validation performance and/or (b) speed of evaluation/ease of hyperparameter tuning. Without ensembling, we would typically pick the algorithms that result in the best cross-validation performance, but as we were ensembling through stacking, we decided to keep any method that might provide additional information that individual model lacks.

The following table shows the classifiers used for stacking along with the parameters used and the dataset used:


| Algorithm | Parameters | Dataset |
|---|---|---|
| Bagging | max\_features: 0.3945<br>n_estimators: 435 | `numerical + categorical slim` |
| Extra Trees | min\_samples\_leaf: 2<br>n_estimators: 99<br>min\_samples\_split: 3<br>max\_features: 1611<br>max\_depth: None} | `numerical + categorical`  |
| Random Forests | min\_samples\_leaf: 1<br> n\_estimators: 77<br>min\_samples\_split: 2<br> random\_state: 1<br> max\_features: 771<br>max\_depth: None | `numerical + categorical`  |
| Logistic Regression | penalty: l1<br>C: 0.9029677391429398 | `numerical + categorical`  |
| *k*NN | n\_neighbors in {1,2,4,8,16,32} | `numerical + categorical slim`  |
| Neural Networks | hidden\_layer_sizes: 900 | `numerical + categorical slim`  <br>

Each of the hyperparameter-tuned classifier was added into the stacking workflow and evaluated as outlined on Predictor Evaluation. Predictor evaluation. Base classifiers were added or removed from the overall stacking workflow based on whether the addition of the classifier improved the 10-fold cross-validation on step 5 of the stacking process.

## Result Evaluation
We did not have a specific hold-out set, but we estimated error using 5- or 10-fold cross-validation. Among the base classifiers, bagging resulted in the best 5-fold cross-validation accuracy of 0.9482. After combining other models using stacking, the best 10-fold cross-validation accuracy obtained was with 0.9491 using bagging classifier as the final meta-classifier.

| Leaderboard | Score   |
| ----------- | ------- |
| Public      | 0.95434 |
| Private     | 0.95345 |

Results obtained with the above methods had a public score of 0.95434 on Kaggle and a private score of 0.95345, good for #2 positions on both leaderboards. The results for our cross-validation were lower since only 80% of the data was used for prediction for the 5-fold cross-validation step. In terms of the difference between the public and the private set, we used only our own cross-validation estimates to select our submission as opposed to the public score, so we predict it is the inherent random nature of the dataset as opposed to over-fitting concerns.
