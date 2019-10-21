# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:00:55 2019

@author: black
#TRY : EXTRACTION, SELECTION, MANIFOLD, DECOMPOSITION modules from scikit-learn

"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor as GBR
from sklearn.linear_model import LassoCV, Lasso
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SS
from sklearn.impute import SimpleImputer as SImp
import copy

data = pd.read_csv('ames_train.csv' , index_col=['Id'])

targets = data.SalePrice
#data.drop(columns=pd.DataFrame(targets), axis=1, inplace=True)

#data.info()
#Use functions for some operations later
numeric = [x for x in data.columns if data[x].dtype == 'int64' or data[x].dtype == 'float64']
numeric = data.loc[:,numeric]
numeric_sc = copy.deepcopy(numeric)
categorical = [x for x in data.columns if data[x].dtype == 'object']
categorical = data.loc[:,categorical].astype('category')

rows = categorical.shape[0]
#Numeric data has very few columns missing very few items (max=17%) so ignored

missers = categorical.isna().sum() / rows
missers =  missers[missers > 0.1].sort_values()*100
missers.plot(kind='bar', title='Percent of Missing Entries'); plt.show()

categorical.drop(columns=missers.index.values, axis=1, inplace=True)
data.drop(columns=missers.index.values, axis=1, inplace=True)


def to_categorical(data):
    """
    Returns all categorical dtypes from all object ones for efficiency reasons
    """
    categorize_label = lambda x: x.astype('category')
    categorical_feature_mask = data.dtypes == object
    categorical_columns = data.columns[categorical_feature_mask].tolist()
    LABELS = categorical_columns
    #Convert df[LABELS] to a categorical type
    data[LABELS] = data[LABELS].apply(categorize_label, axis=0)
    #print(data[LABELS].dtypes)
    #print(data.info())
    return data

to_categorical(data); 

#1. TSNE - Visualization mostly , need to couble heck it works as expected
t = TSNE(learning_rate=50)
tsne_features = t.fit_transform(numeric_sc)
data['x'] = tsne_features[:, 0]
data['y'] = tsne_features[:, 1]


#2. PCA - Need Standardization first as PCA uses variances, and NO nans
#feature scaling  should be done before imputation (found via empirical search)
scaler = SS()
numeric_sc[numeric_sc.columns] = scaler.fit_transform(numeric_sc[numeric_sc.columns]) 
imp = SImp(strategy='median')
numeric_sc[numeric_sc.columns] = imp.fit_transform(numeric_sc[numeric_sc.columns]) 
assert(numeric_sc.isna().sum().sum() == 0)

#use fit_Transform when using the components (plot) - else use just fit
pca = PCA() 
pca.fit(numeric_sc)
#print(pca.explained_variance_ratio_.cumsum()) 


pc_df = pca.fit_transform(numeric_sc)
#pc_df = pd.DataFrame(data=pc_df, columns=numeric.columns, index=numeric.index)
categorical['PC 1'] = pc_df[:, 0]
categorical['PC 2'] = pc_df[:, 1]
vectors = pca.components_.round(2)
#print('PC 1 effects = ' + str(dict(zip(data.columns, pc_df[0]))))
#print('PC 2 effects = ' + str(dict(zip(data.columns, pc_df[1]))))

# Use the Type feature to color the PC 1 vs PC 2 scatterplot
#sns.scatterplot(data=categorical, x='PC 1', y='PC 2', hue='OverallQual')
#plt.show()

seed = 10

#3.Feature selection Algos : Forest, Lasso, GradBoost (only on numeric)
#3.1 RF only
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
numeric[numeric.columns] = imp.fit_transform(numeric[numeric.columns]) 

X = numeric.iloc[:, 0:-3]
X_sc = numeric_sc.iloc[:, 0:-1]
X_train, X_test, y_train, y_test = train_test_split(X_sc, targets, test_size=0.1,
                                                    random_state=seed)
rf = RFR(n_estimators=250)
rf.fit(X_train, y_train)
mse = MSE(y_test, rf.predict(X_test))
feat_imp = dict(zip(X.columns, rf.feature_importances_.round(2)))
feat_imp_df = pd.DataFrame(list(feat_imp.items()), columns=['Feature', 'Importance'] )
feat_imp_df.index = feat_imp_df.Feature.values
feat_imp_df.drop(columns=['Feature'])
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).iloc[0:10]
feat_imp_df.plot(kind='bar', title='Feature Importances: Top 10'); plt.show()
# Print accuracy
print("{} RF RMSE on test set.".format(mse**0.5))
print("{0:.1} RF R^2 on test set.".format(rf.score(X_test, y_test)))

rf_mask = rf.feature_importances_ > 0.10
#reduced_X = X.loc[:, rf_mask]

#3.2 RF + RFE

rfe = RFE(estimator=rf, n_features_to_select=5, step=2, verbose=0)
rfe.fit(X_train, y_train)
rfe_mask = rfe.support_ 
print("{0:.1} RF RFE R^2 on test set.".format(rfe.score(X_test, y_test)))
mse = MSE(y_test, rfe.predict(X_test))
print("{} RF RFE RMSE on test set.".format(mse**0.5))


#3.3 LassoCV
lasso_CV = LassoCV(n_alphas=250, cv=4, n_jobs=2)
lasso_CV.fit(X_train, y_train)
lcv_mask = lasso_CV.coef_ != 0
print('{} features out of {} selected'.format(sum(lcv_mask), len(lcv_mask)))
lasso_CV_coefs =  dict(zip(lasso_CV.coef_.round(4), X_sc.columns))
print("{0:.1} LassoCV R^2 on test set.".format(lasso_CV.score(X_test, y_test)))
mse = MSE(y_test, lasso_CV.predict(X_test))
print("{} LassoCV RFE RMSE on test set.".format(mse**0.5))

#3.4 Gradient Boosting
gbr = GBR(n_estimators=250)
rfe = RFE(estimator=gbr, n_features_to_select=5, step=2, verbose=0)
rfe.fit(X_train, y_train)
gbr_mask = rfe.support_ 
print("{0:.1} GBR RFE R^2 on test set.".format(rfe.score(X_test, y_test)))
mse = MSE(y_test, rfe.predict(X_test))
print("{} GBR RFE RMSE on test set.".format(mse**0.5))

votes = np.sum([lcv_mask, rf_mask, gbr_mask, rfe_mask], axis=0)
print(votes)
meta_mask = votes >= 2
X_reduced = X_sc.loc[:, meta_mask]
print(X_reduced.columns)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, targets, test_size=0.1,
                                                    random_state=seed)

#Hyper tuning with reduced (numeric) cataset SVM gave bad results? Need to try grid search 
from hyperopt import fmin, hp, tpe      #,Trials, space_eval
space = {'gamma': hp.uniform('gamma', 0, 10),'C': hp.uniform('C', 0, 10),}
         'kernel': hp.choice('kernel', ['linear', 'rbf'])}

def objective(params):
    params = {'gamma': params['gamma'],
              'C':params['C'] }
    svm = SVR(**params)
    best_score = cross_val_score(svm, X_train, y_train, scoring='neg_mean_squared_error', cv=4, n_jobs=2).mean()
    loss = 1 - best_score
    return loss

## Run the algorithm
best = fmin(fn=objective,space=space, max_evals=100, rstate=np.random.RandomState(42), algo=tpe.suggest)
print(best)




