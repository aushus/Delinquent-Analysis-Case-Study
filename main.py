import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library to split data
from sklearn.model_selection import train_test_split

# Libraries to build decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV

# To perform statistical analysis
import scipy.stats as stats

# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
)

# Library to suppress warnings or deprecation notes
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Loan_Delinquent_Dataset.csv")
# copying data to another varaible to avoid any changes to original data
loan = data.copy()

print(loan.head())
print(loan.shape)
print(loan.info())
#Checking for missing values
print(loan.isnull().sum())

# checking for duplicate values
data.duplicated().sum()