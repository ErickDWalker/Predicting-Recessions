### Using Recession Predictions to Improve Asset Allocation Strategies
--
### Overview
---
This program aims to predict the occurrence of economic recessions 12-months in the future by training a classification model on past macroeconomic data. Scores output by this model are then used to develop simple trading rules that aim to shift a theoretical investor's portfolio out of the stock market before recessions - and presumably stock market declines - occur. The stock market in this case is the S&P 500, and any money not invested in the market at any given time is assumed to be invested in 10-year U.S. Treasuries.

### Data
---
The independent variables being used to train the model are the yield curve, expressed as the difference between yields on 10-yr and 3-month US Treasury securities, the Civilian Unemployment Rate (U3), and the NFCI Nonfinancial Leverage subindex. These variables are aggregated by month, and then smoothed using a 3-month SMA. The dependent variable is a recession indicator, with one (1) representing the occurrence of recession, and zero (0) the absence of recession. This indicator is shifted 12-months into the past as stock market declines do not perfectly coincide with the onset of recessions, and the aim here is to give investors sufficient advanced warning to lower their equity allocation. All data used to build the recession prediction model is obtained from the Federal Reserve Bank of St. Louis Economic Database (FRED), while S&P500 return data is drawn from Professor Robert Shiller's publically available dataset. 

**Sources:**
1. 10YR - 3M Treasury spread. (T10Y3M column, Frequency = Daily):  https://fred.stlouisfed.org/series/T10Y3M 
2. Recession Indicator (USREC column in data): https://fred.stlouisfed.org/series/USREC 
3. Unemployment rate (UNRATE column, Frequency = Monthly):  https://fred.stlouisfed.org/series/UNRATE
4. NFCI Nonfinancial Leverage Subindex (Frequency = Weekly): https://fred.stlouisfed.org/series/NFCINONFINLEVERAGE
5. 10-yr constant maturity rate (DGS10): https://fred.stlouisfed.org/series/DGS10
6. S&P500 data: http://www.econ.yale.edu/~shiller/data.htm
7. Consumer Price Index (CPI): https://fred.stlouisfed.org/series/CPIAUCSL

### Results
---
In selecting a model, I sought to choose a classifier and associated hyperparameters that maximized a given model's F_Beta score. Beta in this case was the ratio of the S&P500's mean monthly returns during recessions to the same index's mean monthly returns during expansions. My aim in using this metric was to balance the goal of shifting out of the market before a recession hits with the desire to remain invested during the majority of the market’s uptrends. Out of the models I tested, Logistic Regression performed best in this regard.

After choosing a model I moved onto using them to create rules to get investors out of the stock market sufficiently ahead of economic downturns. To do that I ran a number of simulations (for loops) on a hypothetical portfolio, where each simulation adjusts the values of three variables to find the combination of values that yields the maximum portfolio return over the training time frame. These variables are:
1. The model score at which to lower the portfolio's equity allocation
2. The weights to shift the portfolio into once that model score is hit, and
3. The model score at which to resume the portfolio's basline asset allocation 

Other than when the rules counsel a shift in asset allocation, portfolio weights are kept constant, and the default test case for developing these rules was a classic 60 | 40 (stocks | bonds) portfolio, though any mix should yield similar rules. Stated formally, the trading rules which resulted in the highest portfolio returns were: Shift ouf of the stock market into a 10 | 90 (stocks | bonds) allocation once the model score hits 0.70, and hold that allocation until the model score reaches 0.10 or lower (at which point the portfolio returns to its baseline 60 | 40 allocation).
