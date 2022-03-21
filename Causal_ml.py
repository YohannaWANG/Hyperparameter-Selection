from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.inference.meta import BaseRRegressor
from xgboost import XGBRegressor
from causalml.dataset import synthetic_data

y, X, treatment, tau, b, e = synthetic_data(mode=1, n=1000, p=5, sigma=1.0)

"""
Split data into train (70%) and test (30%)
"""
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test, T_train, T_test = train_test_split(X, y, treatment, test_size=0.20, random_state=42)
n = X_test.shape[0]
XA_train = np.concatenate((X_train, T_train.reshape(-1,1)), axis=1)
XA_test = np.concatenate((X_test, T_test.reshape(-1,1)), axis=1)
Xa_one = np.concatenate((np.ones(shape=(n, 1)), X_test), axis=1)
Xa_zero = np.concatenate((np.zeros(shape=(n, 1)), X_test), axis=1)
"""
Models (To be optimized using neural networks)
"""
def model_one():
    # Use linear regression for E[Y| X, A]
    from sklearn import linear_model
    reg = linear_model.LinearRegression().fit(XA_train, Y_train)

    # Use logistic regression \pi(A | X)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0).fit(X_train, T_train)
    return reg, clf

"""
Summary: Now we have two models:
    k1. model 'reg' for E[Y| X, A] (Through Linear Regression)
    l1. model "clf" for \pi[A | X] (Through Logistic Regression)
In our next step, given the trained models, we run prediction use test data
"""
def weird_division(n, d):
    return n / d if d else 0

def ATE(X, T, Y, XA, Xa_one, Xa_zero):

    reg, clf = model_one()
    """ Expand dim for all data to facilitate computing"""
    X = np.expand_dims(X, axis=0)
    XA = np.expand_dims(XA, axis=0)
    Xa_zero = np.expand_dims(Xa_zero, axis=0)
    Xa_one = np.expand_dims(Xa_one, axis=0)
    ate = weird_division((-1)**(1-T), clf.predict(X))*(Y - reg.predict(XA)) + (reg.predict(Xa_one) - reg.predict(Xa_zero))
    return ate

ate_all = []
for i in range(X_test.shape[0]):
    ate = ATE(X_test[i], T_test[i], Y_test[i], XA_test[i], Xa_one[i], Xa_zero[i])
    ate_all.append(ate.astype(float))
ate_dml = np.average(ate_all)
print('Average treatemnt effect using Double machine learning', ate_dml)


lr = LRSRegressor()
ate_lr, lb_lr, ub_lr = lr.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Linear Regression): {:.2f} ({:.2f}, {:.2f})'.format(ate_lr[0], lb_lr[0], ub_lr[0]))

xg = XGBTRegressor(random_state=42)
ate_xg, lb_xg, ub_xg = xg.estimate_ate(X, treatment, y)
print('Average Treatment Effect (XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(ate_xg[0], lb_xg[0], ub_xg[0]))

nn = MLPTRegressor(hidden_layer_sizes=(10),
                 learning_rate_init=.1,
                 early_stopping=True,
                 random_state=42)
ate_nn1, lb_nn1, ub_nn1 = nn.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(ate_nn1[0], lb_nn1[0], ub_nn1[0]))

nn2 = MLPTRegressor(hidden_layer_sizes=(10, 10),
                 learning_rate_init=.1,
                 early_stopping=True,
                 random_state=42)
ate_nn2, lb_nn2, ub_nn2 = nn2.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(ate_nn1[0], lb_nn2[0], ub_nn2[0]))

nn3 = MLPTRegressor(hidden_layer_sizes=(10, 10, 10),
                 learning_rate_init=.1,
                 early_stopping=True,
                 random_state=42)
ate_nn3, lb_nn3, ub_nn3 = nn3.estimate_ate(X, treatment, y)
print('Average Treatment Effect (Neural Network (MLP)): {:.2f} ({:.2f}, {:.2f})'.format(ate_nn3[0], lb_nn3[0], ub_nn3[0]))


xl = BaseXRegressor(learner=XGBRegressor(random_state=42))
ate_xl, lb_xl, ub_xl = xl.estimate_ate(X, treatment, y, e)
print('Average Treatment Effect (BaseXRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(ate_xl[0], lb_xl[0], ub_xl[0]))


rl = BaseRRegressor(learner=XGBRegressor(random_state=42))
ate_rl, lb_rl, ub_rl = rl.estimate_ate(X=X, p=e, treatment=treatment, y=y)
print('Average Treatment Effect (BaseRRegressor using XGBoost): {:.2f} ({:.2f}, {:.2f})'.format(ate_rl[0], lb_rl[0], ub_rl[0]))

"""
Summary of the estimators:
1. ate_dml: double machine learning
2. ate_lr: linear regression
3. ate_xg: XGBoost
4. ate_nn1: one hidden layer neural network (10 hidden units)
5. ate_nn2: two hidden layer neural network (10, 10 hidden unit)
6. ate_nn3: three hidden layer neural network (10, 10, 10 hidden unit)
7. ate_xl: BaseXRegressor using XGBoost
8. ate_rl: BaseRRegressor using XGBoost
"""