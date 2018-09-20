from LoadData import importData
from stratifiedTrainTestSplit import stratifiedTrainTestSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from meature import measure
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier
# from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree
import os

os.environ["PATH"] += os.pathsep + r'D:\软件\Graphviz\install\bin'
############# import data
data_path = 'data/artery.xlsx'
Seed = 45
data = importData(data_path,'data_for_python',Seed)
print(data.head())

############# train test split
X_train, X_test, y_train, y_test = stratifiedTrainTestSplit(data, Seed)
print('the number of 1 in training set:%0.2f, the percentage of 1 in test set:%.2f'%(float(len(X_train[y_train ==0]))/ len(X_train),float(len(X_test[y_test ==0]))/ len(X_test)))
# print(X_train.head(),X_test.head(), y_train.head(), y_test.head())
print(float(len(data[data['Label'] == 0]))/len(data))
# print(y_test)
X_train = X_train.drop(['肝动脉总体'], axis = 1)
test_id = X_test['肝动脉总体']
print('the first 5 test number is:', test_id[:5])
X_test = X_test.drop(['肝动脉总体'], axis = 1)
# print(X_train.head())
# print(X_test.head())
print(len(y_test[y_test == 1]))

############## min-max scaling
scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
print(X_train[:3])
print(X_test[:3])
print(y_train[:3])
print(y_test[:3])
print('training group:',len(X_train))
print('testing group:', len(X_test))

############feature selection
cv = StratifiedShuffleSplit(n_splits = 5, test_size= 0.2, random_state = Seed)
estimator = SVC(probability=True, kernel = 'linear')
selector = RFECV(estimator, step=1, cv=cv, scoring = 'roc_auc')
selector = selector.fit(X_train, y_train)
print('selector.ranking_:', selector.ranking_)
print('selector.n_features_:', selector.n_features_)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
print(X_train[:1], X_test[:1])

###############svm with kernel
learning_algo =SVC(probability=True,class_weight = 'balanced')
search_space = [{'kernel': ['linear'],'C':[10,20.0,50,55,60,65,70,80,100.0,500]},{'kernel':['rbf'],'C':[0.1, 1.0, 10.0,30,50, 80,100.0,200,500,700,800,1000.0],'gamma':[0.0001,0.001,0.005,0.01,0.05,0.1]}]
# # search_space = [{'kernel': ['linear'],'C':np.logspace(1,2,5,100)}]
#search_space = [{'kernel':['rbf'],'C':[0.1, 1.0, 10.0, 100.0,1000.0],'gamma':[0.0001,0.001,0.01,0.1]}]
# search_space = [{'kernel':['rbf'],'C':np.logspace(1,2.3,10),'gamma':np.logspace(-2,1,10)}]

############ ada boost
# learning_algo = AdaBoostClassifier(DecisionTreeClassifier(class_weight= 'balanced'))
# search_space = [{'n_estimators':[50,60,70,80,150,200,300,400],'learning_rate':[0.05, 0.06, 0.07,0.08, 0.5, 1, 1.5]}]
# search_space = [{'n_estimators':[40,80,100,110,120],'learning_rate':np.logspace(-1.5,-1,20)}]
# search_space = [{'n_estimators':[30,40,50,70,80,100,150,180],'learning_rate':[0.03,0.05, 0.1,0.2,0.3,,0.4,,0.5,0.8]}]

############ ann and cv original
# learning_algo = MLPClassifier(solver = 'lbfgs', activation= 'logistic',warm_start=True)
# search_space = [{'hidden_layer_sizes':[(10,),(15,),(2,2),(3,2),(5,2),(2,3),(3,3),(4,3)],'alpha':[0.001,0.005, 0.01,0.02,0.03, 0.1]}]
# # # search_space = [{'hidden_layer_sizes':[(3,),(5,),(10,),(15,),(25,),(2,2),(3,3),(4,4)],'alpha':[0.001,0.005, 0.01,0.02, 0.03, 0.1]}]

# X_train = X_train.drop(['length'], axis = 1)
# X_train = X_train.reshape(-1,1)

# X_test = X_test.drop(['length'], axis = 1)
# X_test = X_test.reshape(-1,1)

############ Decision tree
'''
learning_algo = DecisionTreeClassifier()
search_space = [{'max_depth':[2,3,4,5,6,7,8,9]}]

########## calculate feature importance when max_depth calculated by cv
clf = DecisionTreeClassifier(max_depth= 2)
clf.fit(X_train, y_train)
feature_importance = clf.feature_importances_
print('feature importance:', feature_importance)

########## draw the trained tree
dot_data = tree.export_graphviz(clf, out_file = None, class_names= ['0', '1'])
graph = graphviz.Source(dot_data)
graph.render('PHCC')
'''

############ cv
cv = StratifiedShuffleSplit(n_splits = 30, test_size= 0.3, random_state = Seed)
gridsearch = GridSearchCV(learning_algo, param_grid= search_space, refit = True, cv = cv, scoring = 'roc_auc')
# gridsearch = GridSearchCV(learning_algo, param_grid= search_space, refit = True, cv = cv)
gridsearch.fit(X_train, y_train)

cv_performance = gridsearch.best_score_
print('Best parameter: %s' %str(gridsearch.best_params_))
print('cv_performance:',cv_performance)
measurement2 = measure(gridsearch, X_train, X_test, y_train, y_test)
print('svm with kernel, test_accuracy:%.2f, train_auc: %.2f, test_auc:%.2f' %(measurement2[1],measurement2[4],measurement2[5]))
cf_test = confusion_matrix(y_test, gridsearch.predict(X_test))
cf_train = confusion_matrix(y_train, gridsearch.predict(X_train))
print('test confusion matrix:')
print(cf_test)
print('train confusion matrix:')
print(cf_train)
# print('cutoff',gridsearch.intercept_)


