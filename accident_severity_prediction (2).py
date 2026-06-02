# -- coding: utf-8 --
# Accident Severity Prediction Project
# Extracted from Jupyter Notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv(r"C:\Users\MSS\Downloads\accidents dataset\RTA Dataset.csv")
df.head(3)

df.columns
df.info()
df.isnull().sum()

# ─────────────────────────────────────────
# EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────
print(df['Accident_severity'].value_counts())

df['Accident_severity'].value_counts().plot(
    kind='bar',
    color=['green', 'orange', 'red']
)
plt.title('Accident Severity Distribution')
plt.xlabel('Accident Severity')
plt.ylabel('Count')
plt.show()

df['Educational_level'].value_counts().plot(kind='bar')
plt.title('Educational Level Distribution')
plt.xlabel('Educational Level')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Educational_level', hue='Accident_severity')
plt.xticks(rotation=45, ha='right')
plt.title('Accident Severity by Educational Level')
plt.tight_layout()
plt.show()

print(df['Road_surface_type'].value_counts())
plt.figure(figsize=(10, 5))
sns.countplot(x='Road_surface_type', hue='Accident_severity', data=df,
              palette=['green', 'orange', 'red'])
plt.title('Accident Severity by Road Surface Type')
plt.xlabel('Road Surface Type')
plt.xticks(rotation=60)
plt.show()

print(df['Road_surface_conditions'].value_counts())
plt.figure(figsize=(6, 5))
sns.countplot(x='Road_surface_conditions', hue='Accident_severity', data=df,
              palette=['green', 'orange', 'red'])
plt.title('Accident Severity by Road Surface Conditions')
plt.xlabel('Road Surface Conditions')
plt.xticks(rotation=60)
plt.show()

# Pivot table
pivot_df = pd.pivot_table(data=df, index='Road_surface_conditions',
                          columns='Accident_severity', aggfunc='size', fill_value=0)
fatal_df = pivot_df.copy()
fatal_df['sum_of_injuries'] = (fatal_df['Fatal injury']
                               + fatal_df['Serious Injury']
                               + fatal_df['Slight Injury'])
print(fatal_df)

fatal_df_dry  = (fatal_df.loc['Dry'] / fatal_df.loc['Dry', 'sum_of_injuries']) * 100
fatal_df_Snow = (fatal_df.loc['Snow'] / fatal_df.loc['Snow', 'sum_of_injuries']) * 100
fatal_df_Wet  = (fatal_df.loc['Wet or damp'] / fatal_df.loc['Wet or damp', 'sum_of_injuries']) * 100

df.groupby('Road_surface_conditions')['Accident_severity'].count()

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = df['Time'].dt.hour

obj_cols  = [col for col in df.columns if df[col].dtype == 'object']
obj_cols2 = [col for col in obj_cols if col != 'Accident_severity']

new_df = df.copy()
new_df['Hour_of_Day'] = new_df['Time'].dt.hour
new_df = new_df.drop('Time', axis=1)

def count_plot(col):
    new_df[col].value_counts()
    plt.figure(figsize=(15, 5))
    sns.countplot(x=col, hue='Accident_severity', data=new_df,
                  palette=['green', 'orange', 'red'])
    plt.xlabel(f'{col}')
    plt.xticks(rotation=60)
    plt.show()

for col in obj_cols2:
    count_plot(col)

sns.displot(x='Hour_of_Day', hue='Accident_severity', data=new_df,
            palette=['green', 'orange', 'red'], multiple='stack', height=5, aspect=3)
plt.show()

# ─────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────
features = [
    'Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties',
    'Area_accident_occured', 'Types_of_Junction', 'Age_band_of_driver',
    'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation',
    'Type_of_vehicle', 'Driving_experience', 'Service_year_of_vehicle',
    'Type_of_collision', 'Sex_of_casualty', 'Age_band_of_casualty',
    'Cause_of_accident', 'Hour_of_Day'
]

cols_to_keep = [
    'Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties',
    'Area_accident_occured', 'Types_of_Junction', 'Age_band_of_driver',
    'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation',
    'Type_of_vehicle', 'Driving_experience', 'Service_year_of_vehicle',
    'Type_of_collision', 'Sex_of_casualty', 'Age_band_of_casualty',
    'Cause_of_accident', 'Hour'
]

features_df = df[cols_to_keep].copy()
features_df.columns = features_df.columns.str.strip()
features_df[features_df.select_dtypes('object').columns] = \
    features_df.select_dtypes('object').fillna('Unknown')
features_df.rename(columns={'Hour': 'Hour_of_Day'}, inplace=True)

target = new_df['Accident_severity']
X = features_df[features]
y = target

# ─────────────────────────────────────────
# ENCODING & SMOTE BALANCING
# ─────────────────────────────────────────
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoded_df = pd.DataFrame(oe.fit_transform(X), columns=X.columns)
print(encoded_df.shape)

lb = LabelEncoder()
lb.fit(y)
y_encoded = lb.transform(y)
print("Encoded labels:", lb.classes_)
y_en = pd.Series(y_encoded)

mi_calc = mutual_info_classif(encoded_df, y_en, random_state=42)
mi_df = pd.DataFrame({'Columns': encoded_df.columns, 'MI_score': mi_calc})
print(mi_df.sort_values(by='MI_score', ascending=False).head(15))

fs = SelectKBest(chi2, k=50)
X_new = fs.fit_transform(encoded_df, y_en)
cols = fs.get_feature_names_out()
fs_df = pd.DataFrame(X_new, columns=cols)
print(fs_df.shape)

smote = SMOTE(random_state=42)
X_n, y_n = smote.fit_resample(fs_df, y_en)
print(X_n.shape, y_n.shape)
print(y_n.value_counts())

# ─────────────────────────────────────────
# DATA SPLITTING
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, test_size=0.2, random_state=42)

# ─────────────────────────────────────────
# RANDOM FOREST
# ─────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5,
                            class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)

print("Train score:", rf.score(X_train, y_train))
print(classification_report(y_test, predict))

cm1 = confusion_matrix(y_test, predict)
sns.heatmap(cm1, annot=True)
plt.ylabel('Prediction', fontsize=15)
plt.xlabel('Actual', fontsize=15)
plt.title('Confusion Matrix - Random Forest', fontsize=18)
plt.show()

f1score = f1_score(y_test, predict, average='weighted')
print("F1 Score:", f1score)
RF_score = f1score

# ─────────────────────────────────────────
# DECISION TREE
# ─────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier

d_tree = DecisionTreeClassifier(random_state=9)
d_tree.fit(X_train, y_train)
predict_dt = d_tree.predict(X_test)
score_d_tree = d_tree.score(X_test, y_test)
print('Decision Tree Accuracy:', score_d_tree)

print(classification_report(y_test, predict_dt))

cm2 = confusion_matrix(y_test, predict_dt)
sns.heatmap(cm2, annot=True)
plt.ylabel('Prediction', fontsize=15)
plt.xlabel('Actual', fontsize=15)
plt.title('Confusion Matrix - Decision Tree', fontsize=18)
plt.show()

# ─────────────────────────────────────────
# LOGISTIC REGRESSION
# ─────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

lr_reg = LogisticRegression(C=10)
lr_reg.fit(X_train, y_train)
predict_lr = lr_reg.predict(X_test)
score_lr_reg = lr_reg.score(X_test, y_test)
print('Logistic Regression Accuracy:', score_lr_reg)

print(classification_report(y_test, predict_lr))

cm3 = confusion_matrix(y_test, predict_lr)
sns.heatmap(cm3, annot=True)
plt.ylabel('Prediction', fontsize=15)
plt.xlabel('Actual', fontsize=15)
plt.title('Confusion Matrix - Logistic Regression', fontsize=18)
plt.show()

# ─────────────────────────────────────────
# SVM - LINEAR SVC
# ─────────────────────────────────────────
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

svm_linear = OneVsRestClassifier(LinearSVC(C=10, random_state=9, max_iter=2000))
svm_linear.fit(X_train, y_train)
predict_svm = svm_linear.predict(X_test)
svm_score = svm_linear.score(X_test, y_test)
print("SVM Accuracy:", svm_score)

print(classification_report(y_test, predict_svm))

cm4 = confusion_matrix(y_test, predict_svm)
sns.heatmap(cm4, annot=True)
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix - SVM', fontsize=17)
plt.show()

# ─────────────────────────────────────────
# KNN
# ─────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier

knn_b = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_b.fit(X_train, y_train)
predict_knn = knn_b.predict(X_test)
score_knn_b = knn_b.score(X_test, y_test)
print("KNN Accuracy:", score_knn_b)

print(classification_report(y_test, predict_knn))

cm5 = confusion_matrix(y_test, predict_knn)
sns.heatmap(cm5, annot=True)
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix - KNN', fontsize=17)
plt.show()

# ─────────────────────────────────────────
# ALGORITHM COMPARISON
# ─────────────────────────────────────────
Final_model_accuracy = pd.Series(
    data=[RF_score, score_d_tree, score_knn_b, score_lr_reg, svm_score],
    index=['Random Forest', 'Decision Tree', 'KNN', 'Logistic Regression', 'SVM']
)
fig = plt.figure(figsize=(10, 7))
Final_model_accuracy.sort_values().plot.barh()
plt.title('Accuracy of all the Algorithms')
plt.show()

# ─────────────────────────────────────────
# SAVE MODEL FOR DEPLOYMENT
# ─────────────────────────────────────────
import joblib

joblib.dump(rf,       "model.pkl")          # Random Forest model
joblib.dump(oe,       "encoder.pkl")        # OrdinalEncoder for input features
joblib.dump(fs,       "selector.pkl")       # SelectKBest feature selector
joblib.dump(lb,       "label_encoder.pkl")  # LabelEncoder to decode predictions
joblib.dump(features, "features.pkl")       # Feature column names and order

print("Files saved successfully:")
print("  model.pkl          -> Random Forest classifier")
print("  encoder.pkl        -> OrdinalEncoder (encodes input features)")
print("  selector.pkl       -> SelectKBest (selects top 50 features)")
print("  label_encoder.pkl  -> LabelEncoder (decodes predictions to labels)")
print("  features.pkl       -> Feature names list (ensures correct column order)")
print()
print("In your Streamlit app, load them like this:")
print("  import joblib")
print("  model    = joblib.load('model.pkl')")
print("  encoder  = joblib.load('encoder.pkl')")
print("  selector = joblib.load('selector.pkl')")
print("  lb       = joblib.load('label_encoder.pkl')")
print("  features = joblib.load('features.pkl')")
