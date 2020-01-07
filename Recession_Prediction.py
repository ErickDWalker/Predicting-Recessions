""" 
This program will attempt to calculate a probability of recession 12 months in the future by training a model on past 
macroeconomic data. The practical value of such a predictor is not limited to Federal Reserve officials who attempt to
align interest rates with the current phase of the economic cycle. Another envisioned use is for portfolio managers,
who might wish to shift weights of various asset classes to anticipate a decline in economic activity. 
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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

path = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\csv_files\Used_for_Predictor"
filenames = glob.glob(path + "/*.csv")
df = pd.concat((pd.read_csv(filename, index_col = 0, header=0) for filename in filenames), axis = 1, sort = True)

df.index = pd.to_datetime(df.index)
df = df.sort_index()
df["USREC"] = df["USREC"] * 100
df.to_csv(path_or_buf = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\output_files\combined_data_raw.csv")
# print(df.tail(20)) *view data to ensure proper import

df = df.rename(columns={"S&P":"S&P500"})
df.insert(df.columns.get_loc("S&P500")+1,"S&P500_return",df.pct_change(axis=0)["S&P500"])

# 3-month moving averages to smooth out noise
df["T10Y3M_SMA"] = df.T10Y3M.rolling(3).mean() 
df["UNRATE_SMA"] = df.UNRATE.rolling(3).mean()
df = df.dropna(axis=0)
# save cleaned data df.to_csv(path_or_buf = r"Your Path Here.csv")

# Create DataFrame shifted 12-months,  as we aim to forecast a recession one year ahead
shifted_df = df.copy()
shifted_df["USREC"] = shifted_df["USREC"].shift(-12)
shifted_df = shifted_df.dropna(axis=0)
shifted_df.to_csv(path_or_buf = r"C:\Users\erick\OneDrive\Desktop\Work\Python_files\Recession_Indicator\output_files\combined_data_cleaned_shifted.csv")

classification_target = "USREC"
covariate_names = ["T10Y3M_SMA", "UNRATE_SMA"]

classification_outcome = shifted_df[classification_target]
covariate_data = df[covariate_names]
shifted_covariate_data = shifted_df[covariate_names]
 
# print(training_testing_data) *view data to ensure proper cleaning

# Instantiate regression model
logistic_regression = LogisticRegression(solver = "lbfgs")

X_train, X_test, y_train, y_test = train_test_split(shifted_covariate_data, classification_outcome, train_size = 0.5, random_state=1)
logistic_regression.fit(X_train, y_train) #train the model
recession_prob = pd.DataFrame(logistic_regression.predict_proba(covariate_data)) # predict probability of recession on full dataset
df["PROB_REC"] = recession_prob[1].values * 100 # assign probability of recession (1) to new column in df

print("Logistic regression score:", logistic_regression.score(X_test, y_test)) #test accuracy of model
#print("Probability:", logistic_regression.predict_proba(covariate_data)) #prob of data point being in each class
#print("Class Prediction:", logistic_regression.predict(covariate_data)) #which class model assigns data point to

# save cleaned data to csv file:  df.to_csv(path_or_buf = r"Your Path Here.csv")

# Plot data
fig, axs = plt.subplots(2,figsize=(7,9))

axs[0].plot(df.index, df['T10Y3M_SMA'], '--g', label='T10Y3M_SMA')
axs[0].plot(df['UNRATE_SMA'], '--b', label='UNRATE_SMA')
axs[0].set_ylabel("10YR-less-3M & Unemployment (%)")
axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

ax1 = axs[0].twinx()
ax1.plot(df['USREC'], 'lightgray', label='USREC')
ax1.fill_between(df.index,df['USREC'],color="lightgray",alpha=.3)
ax1.plot(df['PROB_REC'], 'maroon', label='PROB_REC')
ax1.set_ylabel("Recession Indicator & Probability (%)", labelpad=0) # to rotate (rotation = 270)

axs[0].xaxis.set_major_locator(mdates.YearLocator(5))
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, prop={'size': 8})

plt.subplots_adjust(hspace=.2)

# Plot S&P500 Returns
axs[1].plot(df.index, (1+df['S&P500_return']).rolling(12).apply(np.prod, raw=True)-1, 'k', label='S&P500')
axs[1].set_ylabel("S&P500 rolling 12-month return")

ax2 = axs[1].twinx()
ax2.plot(df['USREC'], 'lightgray', label='USREC')
ax2.fill_between(df.index,df['USREC'],color="lightgray",alpha=.3)
ax2.axes.get_yaxis().set_ticks([])
axs[1].axhline(y=0, color='r', linestyle='-')

axs[1].xaxis.set_major_locator(mdates.YearLocator(5))
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Save figure: plt.savefig(r"Your Path Here.pdf")
