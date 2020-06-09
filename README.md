# Modeling County-Level Incarceration Rates in the U.S.

## PROBLEM STATEMENT AND METHODOLOGY:

This project attempts to answer the following question: Can changes in jail incarceration rates be predicted at the county-level through modeling of structural conditions such as employment rates, education, and population and demographics dynamics? Using a subset of incarceration rates data between 2008 and 2017 as compiled by the Vera Institute of Justice from several Bureau of Justice Statistics (BJS) data collections, I appended additional county-level data to model using non-parametric, binary classification algorithms. This data included county-level data on population, demographics, migration, education, income, employment, and veterans. The target variable was constructed by comparing the 2017 jail incarceration rate with the average rate from 2008-2016, with the target labels being Increase or Decrease/Same for the 2017 rate. I followed the OSEMiN framework for this project, sourcing the data from the Vera Institute of Justice and the USDA's Economic Research Service for the other county-level data. After cleaning and exploring the data, I implemented several classification models using GridSearchCV for fine tuning of the hyperparameters for each algorithm. The models used include a baseline Decision Tree Classifier, Random Forest Classifier, Extra Trees Classifier, and AdaBoost Classifier. Modelling was conducted on two sets of data - one with incarceration-related features and one without.

## FINDINGS AND RECOMMENDATIONS:

Beginning with the full data, the AdaBoost Classifier achieved the highest performance metric (F1) score of 87% on our test data, which is an 9% increase compared to the Baseline Decision Tree Classifier's F1 score (78%). Across all the models, each algorithm had higher precision and recall when classifying counties with Increased rates, whereas classification of counties with Decreased/Same rates were not nearly as accurate and had lower F1 scores across all models. When it came to modelling the data without any incarceration-related features, the AdaBoost was only slightly better performing in terms of having the highest F1 score (75.9%), but the Random Forest model performed better in terms of the F1 scores per class label. The AdaBoost model performed poorly when it came to predicting counties with decreased/steady rates, with a recall score of just 23% and an F1 score of 34% for this class label (0, Same or Decrease). By comparison, the Random Forest Classifier had a 33% recall and a 44% F1 score on this class label. The top features in the RF model are Percentage of Non-Veterans who are on Disability, population change rates, natural change rates (2000-2010, and 2010-2018), median household income, percentage of the population aged 65 or older in 2010, and percentage employment change 2017-2018.

## Prerequisites
All data used in this project is available in the *Data* folder in this repo. More information on the data sources is provided at the end of this README. 

Packages used for the analysis:
```python
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# set style, size, palette for all plots
sns.set_style("whitegrid") 
sns.set(rc={'figure.figsize':(14,10)})
sns.set_palette('Set2')
plt.rcParams["axes.labelsize"] = 14

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from IPython.display import Image
```

## Key Findings
- In general, the models struggled with accurately classifying counties with decreased/steady rates, resulting in consistently lower f1 scores for this class label. 
- If we include incarceration-related features, we can see an improvement in the overall F1 scores of 87%, which is a little over a 10% increase from our best performing model on the data *without* the incarceration-related features. 
    - However, including incarceration-related features unsurprisingly means that these same features will have the most predictive power, thereby mitigating the overall understanding of how other population and socio-economic conditions interact with jail incarceration rate changes.
- Looking at the model on data *without* incarceration-related features, we can see that county-level data on disability, population change rates and natural change rates, household income, the proportion of elderly in the population, and percentage change in employment are among the top 10 important features in our best performing model.

### Models on Data *with* incarceration-related features
Our Baseline model performance achieved an F1 score of 78%.

**Best performing model based on F1 Score:** AdaBoost Classifier
- AdaBoost achieved an F1 score of 86.4% on the training data and 86.6% on the test data. Precision was 83% for each label, but recall had some differentiation.  Recall for predicting counties with decreased/steady rates was always lower compared to recall for counties with increased rates (72% and 90%).
- Feature importance across these models shows that previous years' incarceration rates were among the most important, along with population change rates for 2010 to 2018 and from 2017 to 2018.

### Models on Data *without* incarceration-related features
The Baseline model achieved an F1 score of 55.45%, with a training F1 score of 64.8% suggesting our model was overfitting to the training data.

**Best Performing Model based on F1 Score:** AdaBoost Classifier, but...
- Our AdaBoost model achieved an F1 score of 75.9%, only slightly better than the Extra Trees and Random Forest F1 scores (both achieved 75.8%). 
- However, the AdaBoost model performed poorly when it came to predicting counties with decreased/steady rates, with a recall score of just 23% and an F1 score of 34% for this class label (0, Same or Decrease). By comparison, the Decision Tree Classifier and Random Forest both had higher recall scores on this class label (59% and 33% respectively), although precision scores for the Decision Tree were quite poor.
- The Random Forest Classifier had almost as high in terms of the overall F1 score, and it had a higher F1 score on predicting class label 0 (Same or Decrease) at 44%.  
- *Given this, Random Forest is our preferred model, showing an overall improvement to the F1 scroe of almost 31% from the baseline Decision Tree Classifier.*
- The top features in the RF model are Percentage of Non-Veterans who are on Disability, population change rates, natural change rates (2000-2010, and 2010-2018), median household income, percentage of the population aged 65 or older in 2010, and percentage employment change 2017-2018.

## OSEMiN Approach & Summary of Steps
**Obtain:**
* Obtain csv of data of interest from Vera and ERS.
* Load in the incarceration trends data and individual county-level datasets on socio-economic features (people, jobs, income, and veterans). 
**Scrub:**
* Incarceration data: Subset for years of interest (2008-2017)
    * Removed Prison Data, keeping only jail data
    * Deal with null values by first replacing nulls with county-level means (average over the 10 years of data), then state-level means
    * Transform the dataframe so that each year (originally provided per row) into columns
* County-level data (income, people, veterans, jobs):
    * Examine vars contained in variable look up table
    * Check datatypes
    * Deal with Null values (replacing nulls with state averages)
* Drop redundant variables
* Merge all dataframes on FIPS
* Final Null value check
**Explore:**
* Create target variable using rule: compare 2017 rate with average rate over previous 9 years of data
* Set aside test set
* Create version of cleaned data without incarceration-related features (for modeling)
* Explore and visualize data from copy of full training data.
* Convert categorical data to dummy variables (in training and test data).
**Model:**
* Detail modelling approach, performance metrics, and feature importance.
* Per each version of the data (with and without incarceration-related features):
    * Build Baseline Decision Tree Classifier, Random Forest Classifier, Extra Trees Classifier, and AdaBoost Classifier.
    * Produce evaluation/model performance metrics for each.
**Interpret:**
* Reproduce performance metrics for all models and interpret results.
* Extract key findings and considerations for future analysis.

## Analysis
All code is available within the jupyter notebook. Below is the code for the top performing models across the two data versions:

### Data with incarceration-related features
```python
# Construct the pipeline
ab_pipe = Pipeline([('ab', AdaBoostClassifier(random_state=42))])

# Set Grid Search Params
ab_param_grid = {
    'ab__n_estimators':[10,100,500,1000], 
    'ab__learning_rate':[1.0, 0.5, 0.1]
}

# Construct grid search
ab_gs = GridSearchCV(estimator=ab_pipe, 
                    param_grid=ab_param_grid, 
                    scoring='f1', 
                    cv=3, return_train_score=True)

# Fit using gridsearch
ab_gs.fit(X_train, y_train)

# Print best f1 score
print('Best f1 score: %.3f'%ab_gs.best_score_)

# Print best parameters
print('\nBest Params:\n', ab_gs.best_params_)
```
```
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=Pipeline(memory=None,
                                steps=[('ab',
                                        AdaBoostClassifier(algorithm='SAMME.R',
                                                           base_estimator=None,
                                                           learning_rate=1.0,
                                                           n_estimators=50,
                                                           random_state=42))],
                                verbose=False),
             iid='warn', n_jobs=None,
             param_grid={'ab__learning_rate': [1.0, 0.5, 0.1],
                         'ab__n_estimators': [10, 100, 500, 1000]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
             scoring='f1', verbose=0)
Best f1 score: 0.864

Best Params:
 {'ab__learning_rate': 0.1, 'ab__n_estimators': 1000}
```

```python
# Predict on Test Set
ab_gs_pred = ab_gs.predict(X_test)

# Generate metrics for model performance
ab_gs_f1_score = f1_score(y_test, ab_gs_pred)
print("AdaBoost Classifier: F1 score on Test Set: {}".format(ab_gs_f1_score))
print("--------------------------------------------------")
print(classification_report(y_test, ab_gs_pred))
```
```
AdaBoost Classifier: F1 score on Test Set: 0.8655569782330346
--------------------------------------------------
              precision    recall  f1-score   support

           0       0.83      0.72      0.77       246
           1       0.83      0.90      0.87       375

    accuracy                           0.83       621
   macro avg       0.83      0.81      0.82       621
weighted avg       0.83      0.83      0.83       621
```

```python
# Build confusion matrix
unique_label = np.unique([y_test, ab_gs_pred])

cmtx = pd.DataFrame(confusion_matrix(y_test, ab_gs_pred, labels=unique_label), 
                   index=['true: {:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]);
sns.set(font_scale=1.25)
sns.heatmap(cmtx, annot=True);
#sns.heatmap(cmtx/np.sum(cmtx), annot=True, fmt='.2%');

plt.title("Confusion Matrix for AdaBoost Classifier", fontsize=20);
```


### Data *without* incarceration-related features
```python
rf2_clf = RandomForestClassifier(random_state=42)

# Make a param grid for the model
rf2_param_grid = {
    'max_features':['sqrt', 'log2'], 
    'n_estimators':[10,100,500,1000],
    'criterion':['gini', 'entropy']
}

# Construct a grid search
rf2_gs = GridSearchCV(estimator=rf2_clf, 
                      param_grid=rf2_param_grid, 
                      cv=3, scoring="f1", return_train_score=True)

# Fit to training data
rf2_gs.fit(X2_train, y2_train)

# Print training accuracy and best parameters
print("Best Training F1 Score: {}".format(rf2_gs.best_score_))
print("Best Params: {}".format(rf2_gs.best_params_))
```
```
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=RandomForestClassifier(bootstrap=True, class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators='warn', n_jobs=None,
                                              oob_score=False, random_state=42,
                                              verbose=0, warm_start=False),
             iid='warn', n_jobs=None,
             param_grid={'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 'log2'],
                         'n_estimators': [10, 100, 500, 1000]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
             scoring='f1', verbose=0)
Best Training F1 Score: 0.7655763162602993
Best Params: {'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 1000}
```
```python
# run the model on the test set
rf2_gs_pred = rf2_gs.predict(X2_test)

# Produce the weighted F1 score
rf2_gs_f1 = f1_score(y2_test, rf2_gs_pred)
print("Random Forest Classifier: F1 score on Test set: {}".format(rf2_gs_f1))
print(classification_report(y2_test, rf2_gs_pred))

# Build confusion matrix
unique_label = np.unique([y2_test, rf2_gs_pred])

cmtx = pd.DataFrame(confusion_matrix(y2_test, rf2_gs_pred, labels=unique_label), 
                   index=['true: {:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]);
sns.set(font_scale=1.25)
sns.heatmap(cmtx, annot=True);
#sns.heatmap(cmtx/np.sum(cmtx), annot=True, fmt='.2%');

plt.title("Confusion Matrix for Random Forest Classifier (w/o incarceration data)", fontsize=20);
```
```
Random Forest Classifier: F1 score on Test set: 0.7580645161290321
              precision    recall  f1-score   support

           0       0.64      0.33      0.44       246
           1       0.67      0.88      0.76       375

    accuracy                           0.66       621
   macro avg       0.65      0.61      0.60       621
weighted avg       0.66      0.66      0.63       621
```



## Roadmap
* Classification models on top n features to see if model performance improves and if limited features could be used for continuous monitoring and predicting.
* Feature engineering to improve model performance and possibly reduce the number of total features.
* Incorporate additional county-level data, including election data and political party distributions.
* Regression analysis to predict the actual rates per county.


## Authors
* Serena Quiroga - *Capstone Project for the Flatiron School's Immersive Data Science Program*


## Acknowledgements
* Thank you to the Vera Institute of Justice for the multi-stage efforts and enormous success in putting togehter the first-in-kind national database of county and jurisdiction level incarceration trends data. 
* Thank you to the USDA ERS for collecting and making available county-level data.
* Thank you to the Flatiron School and to all of the instructors for being tireless sources of support and encouragement.

## Data Sources:
- **Incarceration Rates Data:**
    - Vera Institute of Justice database of county and jurisdiction level incarceration rates from 1970 to 2017: https://github.com/vera-institute/incarceration_trends
    - The Database is comprised of the following Bureau of Justice Statistics (BJS) data collections: the Census of Jails (COJ), which covers all jails and is conducted every five to eight years since 1970, and the Annual Survey of Jails (ASJ), which covers about one-third of jails-and includes nearly all of the largest jails-that has been conducted in non-census years since 1982, and the BJS National Corrections Reporting Program (NCRP) data collection. Vera merged this data to produce a first-in-kind national dataset that can examine both jail and prison incarceration at the county level.
        - "*Incarceration Trends is supported by Google.org, the John D. and Catherine T. MacArthur Foundation Safety and Justice Challenge, and the Robert W. Wilson Charitable Trust.*" http://trends.vera.org/incarceration-rates?data=pretrial

- **County-level data on Population, Jobs, Education:**
    - The United States Department of Agriculture's (USDA) Economic Research Service (ERS): https://catalog.data.gov/dataset/county-level-data-sets
    - The USDA's Atlas of Rural and Small Town America: 
        - "*Data are grouped by topic and reported in four tabs within the spreadsheet: People, Jobs, Income, Veterans, and County Classifications. Each tab includes the County FIPS Code as the first column. The Variable Name Lookup tab allows users to connect the short name for the indicator used as the header in the spreadsheet with the more descriptive title used in the atlas.*" https://www.ers.usda.gov/data-products/atlas-of-rural-and-small-town-america/download-the-data/