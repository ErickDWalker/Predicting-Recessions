### Using Recession Predictions to Improve Asset Allocation Strategies
---
This program aims to predict the occurrence of economic recessions 12-months in the future by training a model on past macroeconomic data. Scores output by this model are then used to develop simple trading rules that aim to shift a theoretical investor's portfolio out of the stock market before recessions - and presumably stock market declines - occur.

The independent variables being used to train the model are the yield curve, expressed as the difference between yields on 10-yr and 3-month US Treasury securities, the Civilian Unemployment Rate (U3), and the NFCI Nonfinancial Leverage subindex. These variables are aggregated by month, and then smoothed using a 3-month SMA. The dependent variable is a recession indicator, with one (1) representing the occurrence of recession, and zero (0) the absence of recession. This indicator is shifted 12-months into the past as stock market declines do not perfectly coincide with the onset of recessions, and the aim here is to give investors sufficient advanced warning to lower their equity allocation. All data used to build the recession prediction model is obtained from the Federal Reserve Bank of St. Louis Economic Database (FRED), while S&P500 return data is drawn from Professor Robert Shiller's publically available dataset. 

**Sources:**
1. 10YR - 3M Treasury spread. (T10Y3M column, Frequency = Daily):  https://fred.stlouisfed.org/series/T10Y3M 
2. Recession Indicator (USREC column in data): https://fred.stlouisfed.org/series/USREC 
3. Unemployment rate (UNRATE column, Frequency = Monthly):  https://fred.stlouisfed.org/series/UNRATE
4. NFCI Nonfinancial Leverage Subindex (Frequency = Weekly): https://fred.stlouisfed.org/series/NFCINONFINLEVERAGE
5. 10-yr constant maturity rate (DGS10): https://fred.stlouisfed.org/series/DGS10
6. S&P500 data: http://www.econ.yale.edu/~shiller/data.htm
7. Consumer Price Index (CPI): https://fred.stlouisfed.org/series/CPIAUCSL
