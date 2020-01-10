#!/usr/bin/env python
# coding: utf-8

# In[43]:


""" 
This program will attempt to calculate a probability of recession 12 months in the future by training a model on past 
macroeconomic data. The practical value of such a predictor is not limited to Federal Reserve officials attempting to
set interest rate policy based on forecasted economic conditions. Another envisioned use is for portfolio managers,
who might wish to shift weights of various asset classes in anticipation of a decline in economic activity.  

The independent variables being used to train the model are the yield curve, expressed as the difference
between the 10-yr and 3-month yields on US Treasury securities, and the Civilian Unemployment Rate. All data used to build 
the predictor is obtained from the Economic Database of the Federal Reserve Bank of St. Louis (FRED).

Sources:
Recession Indicators (USREC column in data): https://fred.stlouisfed.org/series/USREC
Unemployment rate (UNRATE column):  https://fred.stlouisfed.org/series/UNRATE. 
Yield curve: (T10Y3M column):  https://fred.stlouisfed.org/series/T10Y3M
S&P500 prices: http://www.econ.yale.edu/~shiller/data.htm
*Data used to train the model begins March 1982 and ends November 2019

"""

import pandas as pd
import glob
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Combine data files into DataFrame
path = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\csv_files\Used_for_Predictor"
filenames = glob.glob(path + "/*.csv")
df = pd.concat((pd.read_csv(filename, index_col = 0, header=0) for filename in filenames), axis = 1, sort = True)

df.index = pd.to_datetime(df.index)
df = df.sort_index()
df["USREC"] = df["USREC"] * 100
df.to_csv(path_or_buf = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\output_files\combined_data_raw.csv")
# print(df.tail(20)) *view data to ensure proper import

# Create S&P500 return column and smooth X variables
df = df.rename(columns={"S&P":"S&P500"})
df.insert(df.columns.get_loc("S&P500")+1,"S&P500_return",df.pct_change(axis=0)["S&P500"])
df["T10Y3M_SMA"] = df.T10Y3M.rolling(3).mean() # 3-month moving averages to smooth out noise
df["UNRATE_SMA"] = df.UNRATE.rolling(3).mean()
df = df.dropna(axis=0)
df.to_csv(path_or_buf = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\output_files\combined_data_cleaned.csv")

# create separate DataFrame to house shifted USREC indicator
shifted_df = df.copy() 
shifted_df["USREC"] = shifted_df["USREC"].shift(-12) # shift USREC back 12-months - we aim to forecast a recession one year ahead
shifted_df = shifted_df.dropna(axis=0)
shifted_df.to_csv(path_or_buf = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\output_files\combined_data_cleaned_shifted.csv")
df["shifted_USREC"] = shifted_df["USREC"]
df["shifted_USREC"] = df["shifted_USREC"].fillna(0)

classification_target = "USREC"
covariate_names = ["T10Y3M_SMA", "UNRATE_SMA"]
#covariate_names = ["T10Y3M_SMA"]

classification_outcome = shifted_df[classification_target]
covariate_data = df[covariate_names]
shifted_covariate_data = shifted_df[covariate_names]

df.to_csv(path_or_buf = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\output_files\final_data_with_rec_prob.csv")

# view data to ensure proper cleaning
# print(training_testing_data) 

# Instantiate all regression models and classifiers.
logistic_regression = LogisticRegression(solver = "lbfgs") # <---look into best solver method
random_forest_classifier = RandomForestClassifier(max_depth = 4, random_state = 0) # <--- max_depth / random_state?


# In[ ]:





# In[44]:


# Logistic Regression Classifier
X_train, X_test, y_train, y_test = train_test_split(shifted_covariate_data, classification_outcome, train_size = 0.5, random_state=1)
logistic_regression.fit(X_train, y_train) #train the model
recession_pred_LogReg = pd.DataFrame(logistic_regression.predict(X_test))
recession_prob_LogReg = pd.DataFrame(logistic_regression.predict_proba(covariate_data)) # predict probability of recession on full dataset
df["LOGISTIC_PROB_REC"] = recession_prob_LogReg[1].values * 100 # assign probability of recession (1) to new column in df 

#logistic_regression.coef_
#print("Probability:", logistic_regression.predict_proba(covariate_data)) #prob of data point being in each class
#print("Class Prediction:", logistic_regression.predict(covariate_data)) #which class model assigns data point to
#sklearn.feature_selection.f_regression(X, y, center=True)


# In[45]:


# Allow user to enter data points to output recession probability
count = 0
user_input = []
while count < 2:
    if count == 0: 
        user_input.append(float(input("Enter a 10-yr : 3-month Treasury spread:")))
        
    else: 
        user_input.append(float(input("Enter Unemployment Rate:")))
    count += 1
    
input_recession_score = logistic_regression.predict_proba(np.array(user_input).reshape(1,-1))[0,1] * 100    
print("Recession probability under Logistic Regression: ", "%.2f" % input_recession_score, "%")    

rec_prob_12M_ahead = df[df["shifted_USREC"]==100]["LOGISTIC_PROB_REC"]
print("")

mean_12M_recession_score = rec_prob_12M_ahead.mean()
std_dev_12M_recession_score = np.std(rec_prob_12M_ahead)

print("Summary statistics for points in time 12-months prior to recession indicator = 1:")
print(df[df["shifted_USREC"]==100]["LOGISTIC_PROB_REC"].describe())
print("")

print("Std. Deviations recession score is above (below) mean:", "%.2f" % ((input_recession_score-mean_12M_recession_score)/std_dev_12M_recession_score))


# In[46]:


# Confusion Matrix
# [[True Positive, False Positive],
#  [False Negative, True Negative]]
confusion_matrix_Log_Reg = confusion_matrix(y_test, recession_pred_LogReg)
print(confusion_matrix_Log_Reg)
print("Under Logistic Regression there were:")
print(confusion_matrix_Log_Reg[0,0],"True positives")
print(confusion_matrix_Log_Reg[0,1],"False positives")
print(confusion_matrix_Log_Reg[1,1],"True Negatives")
print(confusion_matrix_Log_Reg[1,0],"False Negatives")
print("Logistic regression score:", "%.2f" % (logistic_regression.score(X_test, y_test)*100),"%") #test accuracy of model


# In[47]:


# Function to plot results
def plot_results(df, column_name):
    fig, axs = plt.subplots(2,figsize=(7,9))

    axs[0].plot(df.index, df['T10Y3M_SMA'], '--g', label='T10Y3M_SMA')
    axs[0].plot(df['UNRATE_SMA'], '--b', label='UNRATE_SMA')
    axs[0].set_ylabel("10YR-less-3M & Unemployment (%)")
    #axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax1 = axs[0].twinx()
    ax1.plot(df['USREC'], 'lightgray', label='USREC')
    ax1.fill_between(df.index,df['USREC'],color="lightgray",alpha=0.3)
    ax1.plot(df[column_name], 'maroon', label='PROB_REC')
    ax1.set_ylabel("Recession Indicator & Probability (%)", labelpad=0) # to rotate (rotation = 270)

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, prop={'size': 8})
    plt.subplots_adjust(hspace=.2)

    # Plot S&P500
    axs[1].plot(df.index, (1+df['S&P500_return']).rolling(12).apply(np.prod, raw=True)-1, 'k', label='S&P500')
    axs[1].set_ylabel("S&P500 rolling 12-month return")

    ax2 = axs[1].twinx()
    ax2.plot(df['USREC'], 'lightgray', label='USREC')
    ax2.fill_between(df.index,df['USREC'],color="lightgray",alpha=.3)
    ax2.axes.get_yaxis().set_ticks([])
    axs[1].axhline(y=0, color='r', linestyle='-')
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # plt.savefig(filepath)
    filepath = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\output_files"
    column_name = column_name + r".pdf"
    save_file_path = os.path.join(filepath, column_name)
    plt.savefig(save_file_path)


# In[48]:


plot_results(df,"LOGISTIC_PROB_REC")


# In[49]:


# Formatting for ROC curve
X_test_prob = pd.DataFrame(logistic_regression.predict_proba(X_test))
y_prob = X_test_prob[1]
# y_test_1 = y_test.reset_index()
# y_test_1 = y_test_1["USREC"]
# X_test = X_test.reset_index()
yield_spread = X_test["T10Y3M_SMA"]
unemployment_rate = X_test["UNRATE_SMA"]

# Forming new df for ROC Curve and Accuracy curve
df2 = pd.DataFrame({ "T10Y3M": yield_spread.values, "UNRATE": unemployment_rate.values, 'y_test': y_test.values, 'model_probability': y_prob})
df2 = df2.sort_values('model_probability')

# Creating 'True Positive', 'False Positive', 'True Negative' and 'False Negative' columns 
df2['tp'] = (df2['y_test'] == int(0)).cumsum()
df2['fp'] = (df2['y_test'] == int(1)).cumsum()
total_0s = df2['y_test'].sum()
total_1s = abs(total_0s - len(df2))
df2['total_1s'] = total_1s
df2['total_0s']= total_0s
df2['total_instances'] = df2['total_1s'] + df2['total_0s']
df2['tn'] = df2['total_0s'] - df2['fp']
df2['fn'] = df2['total_1s'] - df2['tp']
df2['fp_rate'] = df2['fp'] / df2['total_0s']
df2['tp_rate'] = df2['tp'] / df2['total_1s']

# Calculating accuracy column
df2['accuracy'] = (df2['tp'] + df2['tn']) / (df2['total_1s'] + df2['total_0s'])

# Deleting unnecessary columns
df2.reset_index(inplace = True)
del df2['total_1s']
del df2['total_0s']
del df2['total_instances']
del df2['index']

#print(df2.head(10))


# In[50]:


#Plot
plt.plot(df2["model_probability"],df2["accuracy"], color = "c")
plt.xlabel("Model Probability")
plt.ylabel("Accuracy")
plt.title("Optimal Cutoff")


# In[51]:


#Calculating AUC
AUC = 1-(np.trapz(df[‘fp_rate’], df[‘tp_rate’]))

#Plotting ROC/AUC graph
plt.plot(df[‘fp_rate’], df[‘tp_rate’], color = ‘k’, label=’ROC Curve (AUC = %0.2f)’ % AUC)

#Plotting AUC=0.5 red line
plt.plot([0, 1], [0, 1],’r — ‘)
plt.xlabel(‘False Positive Rate’)
plt.ylabel(‘True Positive Rate (Sensitivity)’)
plt.title(‘Receiver operating characteristic’)
plt.legend(loc=”lower right”)
plt.show()


# In[52]:


# Decision Tree Regressor
def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_y = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_y)
    return(mae)


# In[53]:


for max_leaf_nodes in [5,50,100,250,500,1000]:
    mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print("Max leaf nodes:",max_leaf_nodes,"          Mean Absolute Error:", "%.2f" % mae)


# In[54]:


#Random Forest Classifier
random_forest_classifier = RandomForestClassifier(max_depth = 4, random_state = 0)
random_forest_classifier.fit(X_train, y_train)
recession_random_forest_prob = pd.DataFrame(random_forest_classifier.predict_proba(covariate_data))
df["RANDOM_FOREST_PROB_REC"] = recession_random_forest_prob[1].values * 100 # assign probability of recession (1) to new column in df

#Calculating the accuracy of the training model on the testing data
recession_pred_Random_Forest = pd.DataFrame(random_forest_classifier.predict(X_test))


# In[55]:


# Confusion Matrix
# [[True Positive, False Positive],
#  [False Negative, True Negative]]
confusion_matrix_Random_Forest = confusion_matrix(y_test, recession_pred_Random_Forest) # compare actual test outcomes to predicted outcomes 
print(confusion_matrix_Random_Forest)
print("Under Logistic Regression there were:")
print(confusion_matrix_Random_Forest[0,0],"True positives")
print(confusion_matrix_Random_Forest[0,1],"False positives")
print(confusion_matrix_Random_Forest[1,1],"True Negatives")
print(confusion_matrix_Random_Forest[1,0],"False Negatives")
print("Random forest classifier score: ", "%.2f" % (random_forest_classifier.score(X_test, y_test) * 100), "%")


# In[56]:


# Plot results of Random Forest 
plot_results(df,"RANDOM_FOREST_PROB_REC")


# In[57]:


def accuracy(estimator, X, y): # Classification
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)

logistic_regression_scores = cross_val_score(logistic_regression, shifted_covariate_data, classification_outcome, cv=10, scoring=accuracy)
random_forest_classification_scores = cross_val_score(random_forest_classifier, shifted_covariate_data, classification_outcome, cv=10, scoring=accuracy)

# print("Logistic Regression Scores:",logistic_regression_scores)
# print("Forest Classification Scores:",forest_classification_scores)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, random_forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

plt.show()

