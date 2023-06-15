# %%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from imblearn.combine import SMOTEENN

# %%
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# %%
df.head(5)

# %%
df.describe()

# %%
df['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);

# %%
100*df['Churn'].value_counts()/len(df['Churn'])

# %%
df['Churn'].value_counts()

# %%
df.info(verbose = True)

# %%
df.drop('customerID',axis='columns',inplace=True)
df.dtypes

# %%
df.TotalCharges.values

# %%
df.MonthlyCharges.values

# %%
pd.to_numeric(df.TotalCharges)

# %%
pd.to_numeric(df.TotalCharges, errors='coerce').isnull()

# %%
df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()]

# %%
df.iloc[488]['TotalCharges']

# %%
df.iloc[488]['TotalCharges']

# %%
df1 = df[df.TotalCharges!= ' ']
df1.shape

# %%
df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()].shape

# %%
df.shape

# %%
df1.dtypes

# %%
df1.Totalcharges = pd.to_numeric(df1.TotalCharges)

# %%
df1.Totalcharges.dtypes

# %%
df1[df1.Churn=='No']

# %%
tenure_Churn_no = df1[df1.Churn=='No'].tenure
tenure_Churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_Churn_yes, tenure_Churn_no], color =['green', 'red'],label=['Churn=Yes','Churn=No'])
plt.legend()

# %%
mc_Churn_no = df1[df1.Churn=='No'].MonthlyCharges
mc_Churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges

plt.xlabel("MonthlyCharges")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")

blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 79, 120, 112, 100]

plt.hist([mc_Churn_yes, mc_Churn_no], rwidth=0.95, color =['green', 'red'],label=['Churn=Yes','Churn=No'])
plt.legend()

# %%
def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column} : {df[column].unique()}')

# %%
print_unique_col_values(df1)

# %%
df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)

# %%
print_unique_col_values(df1)

# %%
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)

# %%
for col in df1:
    print(f'{col}: {df1[col].unique()}') 

# %%
df1['gender'].replace({'Female':1,'Male':0},inplace=True)

# %%
df1.gender.unique()

# %%
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
df2.columns

# %%
df2.sample(4)

# %%
df2.dtypes

# %%
cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

# %%
df2.sample(3)

# %%
for col in df2:
    print(f'{col}: {df2[col].unique()}')

# %%
import seaborn as sns
for i, predictor in enumerate(df2.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=df2, x=predictor, hue='Churn')

# %%
X = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

# %%
X_train.shape

# %%
X_test.shape

# %%
X_train[:10]

# %%
len(X_train.columns)

# %%
import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

# %%
model.evaluate(X_test, y_test)

# %%
yp = model.predict(X_test)
yp[:5]
y_test[:5]

# %%
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# %%
y_pred[:5]

# %%
y_test[:10]

# %%
from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))

# %%
import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# %%
y_test.shape

# %%
round((862+229)/(862+229+137+179),2) 

# %%
round(862/(862+179),2)

# %%
round(229/(229+137),2)

# %%
round(862/(862+137),2)

# %%
round(229/(229+179),2)

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6,min_samples_leaf=8)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# %%
print(classification_report(y_test, y_pred_rf, labels=[0,1]))

# %%
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# %%
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_resampled,y_resampled, test_size=0.2)

# %%
model_smote_rf = RandomForestClassifier(criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)

# %%
model_smote_rf.fit(Xr_train,yr_train)

# %%
y_pred_smote_rf = model_smote_rf.predict(Xr_test)

# %%
print(classification_report(yr_test, y_pred_smote_rf, labels = [0,1]))

# %%
print(confusion_matrix(yr_test, y_pred_smote_rf))

# %%
model_dt = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)

# %%
model_dt.fit(X_train, y_train)

# %%
y_pred = model_dt.predict(X_test)

# %%
y_pred

# %%
print(classification_report(y_test, y_pred, labels=[0,1]))

# %%
print(confusion_matrix(y_test, y_pred))

# %%
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# %%
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_resampled,y_resampled, test_size=0.2)

# %%
model_dt_smote=DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)

# %%
model_dt_smote.fit(Xr_train,yr_train)

# %%
y_pred_smote = model_dt_smote.predict(Xr_test)

# %%
print(classification_report(yr_test, y_pred_smote, labels = [0,1]))

# %%
print(confusion_matrix(yr_test, y_pred_smote))

# %%
import pickle

# %%
filename = 'model.sav'

# %%
pickle.dump(model_smote_rf, open(filename, 'wb'))

# %%
load_model = pickle.load(open(filename, 'rb'))

# %%
load_model.score(Xr_test, yr_test)

# %%



