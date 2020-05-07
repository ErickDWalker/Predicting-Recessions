# Recession_Prediction
---
This program aims to calculate a probability of recession 12 months in the future by training a model on past  macroeconomic data. The independent variables being used to train the model are the yield curve, expressed as the difference between the 10-yr and 3-month yields on US Treasury securities, and the Civilian Unemployment Rate. The original, raw data is monthly, but I then smooth these series using 3-month moving averages. All data used to build  the predictor is taken from the St. Louis Federal Reserve Bank's Economic Database (FRED), while due credit for historical S&P500 returns is given to Yale Professor Robert Shiller: http://www.econ.yale.edu/~shiller/data/ie_data.xls.

*Data used to train the model begins March 1982 and ends December 2019. While the start of this period may appear to conveniently overlap with the beginning of available data for the yield spread, this is merely coincidence. I believe the period from the early 1980s onwards is more representative of economic conditions likely to be seen going forward. The decade leading up to 1982 saw the rise of stagflation, as well as the onset of recessions from historically high levels of unemployment, among other major differences with today's macroeconomy. Given one of the features used to train the model is the unemployment rate (with the underlying thesis that in normal economic conditions recessions tend to start from low unemployment levels, as scarce labor drives up input costs), it would seem appropriate to limit the period studied to the more recent past.
