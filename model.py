import sys
import logging
import pandas as pd
import numpy as np
import scipy

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

log = logging.getLogger('model')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

log.info('Read data...')
train_orig = pd.read_table("data/train.tsv", dtype={'item_condition_id':str, 'shipping':str})

# split category name
def sort_category(dataset):
    cat_nbr = list()
    # sort out all categories
    category_set = set(["NA"])
    n_lvl = 5
    category_lvls = list()
    category_lvls_idx = list()
    category_lvls_name = ["cat_lvl_"+str(x) for x in range(n_lvl)]
    for i, v in dataset["category_name"].iteritems():
        category_lvls_idx.append(i)
        if not pd.isnull(v):
            cate_ls = v.split("/")
            cate_ls = tuple(cate_ls+["NA"]*(n_lvl-len(cate_ls)))
            category_lvls.append(cate_ls)
            cat_nbr.append(len(cate_ls))
            category_set |= set(cate_ls)
        else:
            category_lvls.append(["NA"]*n_lvl)
    category_lvls = pd.DataFrame(category_lvls, index=category_lvls_idx, columns=category_lvls_name)
    return category_lvls, category_set

# modify "category_name"
cg_lvls, cg_set = sort_category(train_orig)
train_mod = pd.concat([train_orig, cg_lvls], axis=1)
train_mod.drop("category_name", axis=1, inplace=True)

# fill missing values
null_idx = np.where(train_mod.isnull().any())[0]
for i in null_idx:
    train_mod.loc[pd.isnull(train_mod.iloc[:,i]), train_mod.columns[i]] = "noname"

# log-transform the target variable
train_mod["pricelog"] = pd.Series(np.log10(train_mod["price"]+1), index=train_mod["price"].index)

# encoding
log.info('Encoding...')
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(train_mod[[
    "item_condition_id", "shipping"]], sparse = True).values)

vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(train_mod["brand_name"])

vect_cat0 = LabelBinarizer(sparse_output=True)
X_cat0 = vect_cat0.fit_transform(train_mod["cat_lvl_0"])

vect_cat1 = LabelBinarizer(sparse_output=True)
X_cat1 = vect_cat1.fit_transform(train_mod["cat_lvl_1"])

vect_cat2 = LabelBinarizer(sparse_output=True)
X_cat2 = vect_cat2.fit_transform(train_mod["cat_lvl_2"])

X = scipy.sparse.hstack((X_dummies, 
                         X_cat0,
                         X_cat1,
                         X_cat2,
                         X_brand)).tocsr()

# train-test split
log.info('Split train-test...')
X_train, X_test, y_train, y_test = train_test_split(X, train_mod['pricelog'], test_size=0.2, random_state=0)

params = {
    'linear regression': {
        'model': LinearRegression(),
        'params': [{'model__fit_intercept': [True]}]
    },
    'sgd regression': {
        'model': SGDRegressor(),
        'params': [{'model__penalty': ['l2']}]
    }
}
model_name = 'linear regression'

# training
log.info('Training...')
pipeline = Pipeline([('model',params[model_name]['model'])])
clf = GridSearchCV(pipeline, params[model_name]['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=1, verbose=10)
#clf = params[model_name]['model']
clf.fit(X_train, y_train)

# predict
log.info('Predict...')
y_pred = clf.predict(X_test)
score = mean_squared_error(y_test, y_pred)
