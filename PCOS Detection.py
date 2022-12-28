import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, svm
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

import sklearn.metrics as metrics
import plotly.express as px

# Importing Dataset
df = pd.read_csv(r"Datasets/PCOS_data.csv")

"""
#### Data properties
print(df) 
# Number of data points
print(len(df))
# Number of features
print(len(df.columns))
# datatypes
print(df.dtypes.unique())
print(df.keys())

print(df.info())
print(df.describe())
"""

# Handling Null values in the data
nulls = []
for col in df:
    if df[col].isnull().sum() != 0:
        nulls.append(col)
        if df[col].dtype == 'float' or df[col].dtype == 'int':
            df[col] = df[col].fillna(value=df[col].mean())
        elif df[col].dtype == object:
            df[col] = df[col].fillna(df[col].mode()[0])
        df = df
print('Columns with null values were:', nulls)
# checking null values in the data
print("Number of nan values in columns: ", df.isnull().sum().sum())


def iqr(data, columns):
    q1, q3 = data[columns].quantile(0.25), data[columns].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


# Getting outlier columns
outlier_columns = []
for col in df:
    if df[col].dtype == 'float' or df[col].dtype == 'int':
        lower, upper = iqr(df, col)
        if (df[col] > upper).any() | (df[col] < lower).any():
            outlier_columns.append(col)
            df[col] = np.where((df[col] > upper) | (df[col] < lower), df[col].mean(), df[col])
print('Outlier columns are :', outlier_columns)

# correlated data
# plt.figure(figsize=(50, 50))
# sns.heatmap(df.corr(), annot=True, linewidth=0.5, linecolor='RED', fmt="1.2f")
# plt.title("Data correlation", fontsize=50)
# plt.show()

# Dealing with categorical values.
# In this database the type objects are numeric values saved as strings.
# So just converting it into a numeric value.
df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"], errors='coerce')
df["II    beta-HCG(mIU/mL)"] = pd.to_numeric(df["II    beta-HCG(mIU/mL)"], errors='coerce')
# print(df.dtypes)

# Assigning the features (X) and target(y)
X = df.drop(["PCOS (Y/N)", "Sl. No", "Patient File No.", "Unnamed: 44"], axis=1)  # dropping out index from features too
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
# print(X)
y = df["PCOS (Y/N)"]

# Splitting the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

comparing_models = pd.DataFrame(columns=('test_accuracy', 'f1_score'))
test_accuracy = []


class Model:
    model = None
    accuracy = 0
    f_score = 0
    pred_model = None
    model_type = None

    def __init__(self, model_type, params, X_train, y_train, X_test, y_test):
        self.model_type = model_type
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        if self.model_type == 'Random Forest Classifier':
            self.model = RandomForestClassifier()
        elif self.model_type == 'Extreme Gradient Boosting':
            self.model = XGBClassifier(objective='binary:logistic', eval_metric="logloss", use_label_encoder=False)
        elif self.model_type == 'Multi Layer Perceptron':
            self.model = MLPClassifier()
        elif self.model_type == 'Support Vector Machine':
            self.model = svm.SVC()
        elif self.model_type == 'AdaBoost Classifier':
            self.model = AdaBoostClassifier(n_estimators=100)
        # Over Sampling
        sm = SMOTE()
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        random = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20,
                                    scoring='roc_auc', n_jobs=2,
                                    cv=skf.split(self.X_train, self.y_train), verbose=2, random_state=42)

        data_model = random.fit(self.X_train, self.y_train)
        self.pred_model = data_model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.pred_model)
        self.f_score = f1_score(y_test, self.pred_model, average="macro")
        print(f'Accuracy for model {self.model_type}:{self.accuracy}')
        print(f'Classification report for {self.model_type}:\n', classification_report(self.y_test, self.pred_model))

    def print_confusion_matrix(self):
        plt.figure(dpi=100)
        plt.title(f"Confusion Matrix for {self.model_type}")
        matrix = confusion_matrix(np.ravel(self.y_test), self.pred_model)
        sns.heatmap(matrix, annot=True)

    def plot_roc_curve(self):
        specificity, sensitivity, threshold = metrics.roc_curve(self.y_test, self.pred_model)
        roc = metrics.auc(specificity, sensitivity)
        plt.figure(dpi=100)
        plt.title(f'ROC curve for {self.model_type}')
        plt.plot(specificity, sensitivity, 'b', label='AUC = %0.2f' % roc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate(sensitivity)')
        plt.xlabel('False Positive Rate(specificity)')
        plt.show()


# RandomForestClassifier
params = {'bootstrap': [True, False],
          'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
          'max_features': ['auto', 'sqrt'],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [2, 5, 10],
          'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
model = Model('Random Forest Classifier', params, X_train, y_train, X_test, y_test)
model.train_model()
test_accuracy.append(model.accuracy)

comparing_models = comparing_models.append(pd.DataFrame({'test_accuracy': [model.accuracy],
                                                         'f1_score': [model.f_score]},
                                                        index=['Random Forest Classifier']))
# print(comparing_models)
comparing_actual_vs_predicted = pd.DataFrame({'actual': y_test, 'Predicted': model.pred_model})
print(comparing_actual_vs_predicted)
model.print_confusion_matrix()
model.plot_roc_curve()

# XGBClassifier
params = {'min_child_weight': [1, 5, 10],
          'gamma': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5],
          'learning_rate': [0.01, 0.03, 0.06, 0.09, 0.15, 0.25, 0.3, 0.4, 0.5],
          'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
          'colsample_bytree': [0.6, 0.8, 1.0],
          'n_estimators': [50, 65, 80, 100, 150, 200, 300, 400, 500],
          'reg_alpha': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5],
          'reg_lambda': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5]}
model = Model('Extreme Gradient Boosting', params, X_train, y_train, X_test, y_test)
model.train_model()
test_accuracy.append(model.accuracy)

comparing_models = comparing_models.append(pd.DataFrame({'test_accuracy': [model.accuracy],
                                                         'f1_score': [model.f_score]},
                                                        index=['Extreme Gradient Boosting']))
# print(comparing_models)
comparing_actual_vs_predicted = pd.DataFrame({'actual': y_test, 'Predicted': model.pred_model})
print(comparing_actual_vs_predicted)
model.print_confusion_matrix()
model.plot_roc_curve()

# MLPClassifier
params = {'solver': ['lbfgs'], 'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
          'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes': np.arange(10, 15),
          'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

model = Model('Multi Layer Perceptron', params, X_train, y_train, X_test, y_test)
model.train_model()
test_accuracy.append(model.accuracy)

comparing_models = comparing_models.append(pd.DataFrame({'test_accuracy': [model.accuracy],
                                                         'f1_score': [model.f_score]},
                                                        index=['Multi Layer Perceptron']))
# print(comparing_models)
comparing_actual_vs_predicted = pd.DataFrame({'actual': y_test, 'Predicted': model.pred_model})
print(comparing_actual_vs_predicted)
model.print_confusion_matrix()
model.plot_roc_curve()

# SVM (SVC)
params = {'C': [0.1, 1, 10, 100, 200, 400, 500], 'gamma': [1, 0.9, 0.1, 0.01, 0.001],
          'kernel': ['rbf', 'poly', 'sigmoid']}

model = Model('Support Vector Machine', params, X_train, y_train, X_test, y_test)
model.train_model()
test_accuracy.append(model.accuracy)

comparing_models = comparing_models.append(pd.DataFrame({'test_accuracy': [model.accuracy],
                                                         'f1_score': [model.f_score]},
                                                        index=['Support Vector Machine']))
# print(comparing_models)
comparing_actual_vs_predicted = pd.DataFrame({'actual': y_test, 'Predicted': model.pred_model})
print(comparing_actual_vs_predicted)
model.print_confusion_matrix()
model.plot_roc_curve()

# Adaboost Classifier
params = {
          'n_estimators': [100, 400, 600, 800]
}

model = Model('AdaBoost Classifier', params, X_train, y_train, X_test, y_test)
model.train_model()
test_accuracy.append(model.accuracy)

comparing_models = comparing_models.append(pd.DataFrame({'test_accuracy': [model.accuracy],
                                                         'f1_score': [model.f_score]},
                                                        index=['AdaBoost Classifier']))

comparing_actual_vs_predicted = pd.DataFrame({'actual': y_test, 'Predicted': model.pred_model})

model.print_confusion_matrix()
model.plot_roc_curve()

# Accuracy and f1_score comparison for all models
print(comparing_models)

# Plot comparison
plot = px.line(comparing_models, x=comparing_models.index, y=['test_accuracy', 'f1_score'])

plot.show()
