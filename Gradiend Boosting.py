import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


df = pd.read_csv("Dataset passenger satisfaction2.csv", sep=";")

df["arrival_delay_minutes"].fillna(df["departure_delay_minutes"], inplace= True)
print(df["arrival_delay_minutes"].isnull().sum())
df["arrival_delay_minutes"] = df["arrival_delay_minutes"].astype("int64")

#Label encoding for Gender: Female = 0 Male = 1
df["gender"].unique()
gender_mapping = {"Female" : 0, "Male" : 1}
df["gender"] = df["gender"].map(gender_mapping)

#Label encoding customer type: disloyal Customer = 0 Loyal Customer = 1
df["customer_type"].unique()
customer_mapping = {"disloyal Customer" : 0, "Loyal Customer" : 1}
df["customer_type"] = df["customer_type"].map(customer_mapping)


#Label encoding Type Travel: Personal Travel = 0 Business = 1
df["type_travel"].unique()
travel_mapping = {"Personal Travel" : 0, "Business travel" : 1}
df["type_travel"] = df["type_travel"].map(travel_mapping)


#Label encoding Class as ordinal variable: Eco  = 0 Eco Plus = 1 Business = 2
df["class"].unique()
class_mapping = {"Eco" : 0, "Eco Plus" : 1, "Business" : 2}
df["class"] = df["class"].map(class_mapping)

#Label encoding Class as ordinal variable: Eco  = 0 Eco Plus = 1 Business = 2
df["satisfaction"].unique()
satisfaction_mapping = {"neutral or dissatisfied" : 0, "satisfied" : 1}
df["satisfaction"] = df["satisfaction"].map(satisfaction_mapping)

#BUILDING THE MODEL
featuresdrop = ["id", "customer_type", "arrival_delay_minutes"]
df.drop(featuresdrop, axis = 1, inplace= True)

X = df.drop("satisfaction", axis = 1).values
Y = df["satisfaction"].values


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
X_train.shape

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=50, learning_rate= 0.3, max_depth= 4, random_state = 0)
gbc.fit(X_train, Y_train)

y_pred_train =gbc.predict(X_train)
Y_pred = gbc.predict(X_test)
Y_pred_proba = gbc.predict_proba(X_test)[::,1]

from sklearn.metrics import accuracy_score 
from sklearn.metrics import log_loss 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics

print("ACCURACY: "+str(accuracy_score(Y_test, Y_pred)))
print("F1 Score: "+str(f1_score(Y_test, Y_pred)))
print("PRECISION: "+str(precision_score(Y_test, Y_pred)))
print("RECALL: "+str(recall_score(Y_test, Y_pred)))

#ROC CURVE
#define metrics
fpr, tpr, _ = metrics.roc_curve(Y_test,  Y_pred_proba)
auc = metrics.roc_auc_score(Y_test, Y_pred_proba)

#create ROC curve
plt.rcParams["figure.figsize"] = (12,9)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title ('Gradient Boosting - Receiver operating characteristic')
plt.legend(loc=4)
plt.show()

# get importance
importance = gbc.feature_importances_

usedfeature = df.drop("satisfaction" ,axis = 1)
gb_importances = pd.Series(importance, index=usedfeature.columns).sort_values(ascending=True)

ax = gb_importances.plot.barh()
plt.rcParams["figure.figsize"] = (14,10)
ax.set_title("Gradient Boosting Classifier Feature Importances")
plt.xlabel('Impurity Based Feature Importance')
ax.figure.tight_layout()

#SHAP
import shap
shap.initjs()


#beeswarm
explainer = shap.TreeExplainer(gbc)
shap_values = np.array(explainer.shap_values(X_train))
shapplot = shap.summary_plot(shap_values, X_train, feature_names=usedfeature.columns)

#Feature importance using permutation
from sklearn.inspection import permutation_importance

result = permutation_importance(
    gbc, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=usedfeature.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
plt.rcParams["figure.figsize"] = (11,8)
ax.set_title("Gradient Boosting Classifier Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
    





