import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


df = pd.read_csv("Dataset passenger satisfaction2.csv", sep=";")

#Computazione dei dati mancanti, assegnare valore departurer delay a mancanti ArrivalDelay
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

df.info()
df.shape


#BUILDING THE MODEL
featuresdrop = ["id", "customer_type", "arrival_delay_minutes"]
df.drop(featuresdrop, axis = 1, inplace= True)


X = df.drop("satisfaction", axis = 1).values
Y = df["satisfaction"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
X_train.shape


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
Y_pred_proba = decision_tree.predict_proba(X_test)[::,1]

from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


print("ACCURACY: "+str(accuracy_score(Y_test, Y_pred)))
print("F1 Score: "+str(f1_score(Y_test, Y_pred)))
print("PRECISION: "+str(precision_score(Y_test, Y_pred)))
print("RECALL: "+str(recall_score(Y_test, Y_pred)))

#Visualize the tree

from sklearn.tree import export_graphviz
dotfile = open("decision_tree.dot", 'w')
export_graphviz(decision_tree, out_file = dotfile, feature_names = df.columns.drop("satisfaction"))
dotfile.close()


# get importance
importance = decision_tree.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
    
    
usedfeature = df.drop("satisfaction" ,axis = 1)   
tree_importances = pd.Series(importance, index=usedfeature.columns).sort_values(ascending=True)
     
#plot importances score = Gini
plt.rcParams["figure.figsize"] = (11,8)
ax = tree_importances.plot.barh()
ax.set_title("Classification Tree Gini Score")
plt.xlabel('Gini Score')
ax.figure.tight_layout()
    
#ROC CURVE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics

#define metrics
fpr, tpr, _ = metrics.roc_curve(Y_test,  Y_pred_proba)
auc = metrics.roc_auc_score(Y_test, Y_pred_proba)

#create ROC curve
plt.rcParams["figure.figsize"] = (12,9)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title ('Classification Tree - Receiver operating characteristic')
plt.legend(loc=4)
plt.show() 
    
 
import shap
shap.initjs()

explainer = shap.TreeExplainer(decision_tree)
shap_values = explainer(X)
shap.summary_plot(shap_values, feature_names = usedfeature.columns)



shap_values = shap.TreeExplainer(decision_tree).shap_values(X)
shap.summary_plot(shap_values, feature_names = usedfeature.columns)

#beeswarm
explainer2 = shap.TreeExplainer(decision_tree)
shap_values2 = np.array(explainer2.shap_values(X_train))
shapplot = shap.summary_plot(shap_values2[1], X_train, feature_names=usedfeature.columns)

#Feature importance using permutation
from sklearn.inspection import permutation_importance

result = permutation_importance(
    decision_tree, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=usedfeature.columns[sorted_importances_idx],
)

ax = importances.plot.box(vert=False, whis=10)
plt.rcParams["figure.figsize"] = (20,8)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
