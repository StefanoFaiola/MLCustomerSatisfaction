
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics


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

#Eseguire standardizzazione per ottenere i coefficienti sulla stessa scala
features = ["age", "flight_distance", "inflight_wifi_service", "departure_arrival_time_convenient",
            "ease_online_booking", "gate_location", "food_drink", "online_boarding", "seat_comfort",
            "inflight_entertainment", "onboard_service", "leg_room_service", "baggage_handling",
            "checkin_service", "inflight_service", "cleaniless", "departure_delay_minutes", "arrival_delay_minutes"]
to_std = df[features]
df[features] = (to_std - to_std.mean())/to_std.std()



#BUILDING THE MODEL
featuresdrop = ["id", "customer_type", "arrival_delay_minutes"]
df.drop(featuresdrop, axis = 1, inplace= True)

X = df.drop("satisfaction", axis = 1).values
Y = df["satisfaction"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

lr = LogisticRegression(penalty = "none")
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)
cm = confusion_matrix(Y_test,Y_pred,labels=[1,0])

#Analizziamo le metriche del nostro modello
from sklearn.metrics import accuracy_score 
from sklearn.metrics import log_loss 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print("ACCURACY: "+str(accuracy_score(Y_test, Y_pred)))
print("F1 Score: "+str(f1_score(Y_test, Y_pred)))
print("PRECISION: "+str(precision_score(Y_test, Y_pred)))
print("RECALL: "+str(recall_score(Y_test, Y_pred)))

importance = lr.coef_[0]
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
    
usedfeature = df.drop("satisfaction" ,axis = 1)
logistic_importances = pd.Series(importance, index=usedfeature.columns).sort_values(ascending=True)  
        
#plot migliore, metodo mdi
plt.rcParams["figure.figsize"] = (9,6)
ax = logistic_importances.plot.barh()
ax.set_title("Logistic Regression Coefficients")
ax.figure.tight_layout()
    
    
#ROC CURVE
#define metrics
y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)

#create ROC curve
plt.rcParams["figure.figsize"] = (9,6)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title ('Logistic Regression - Receiver operating characteristic')
plt.legend(loc=4)
plt.show()

#ROC Curve versione migliore
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(Y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(Y_test, lr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#import statsmodels.api as sm
#X_train_with_constant = sm.add_constant(X_train)
#X_test_with_constant = sm.add_constant(X_test)

#sm_model=sm.Logit(Y_train, X_train_with_constant).fit()

#print(sm_model.summary2())
#print(sm_model.params)

import shap
shap.initjs()
explainer = shap.LinearExplainer(lr, masker=shap.maskers.Impute(data=X_train), feature_names = usedfeature.columns, feature_perturbation="interventional")
shap_values = explainer(X)
shapplot = shap.summary_plot(shap_values)


#Feature importance using permutation
from sklearn.inspection import permutation_importance

result = permutation_importance(
    lr, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=2
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


