# Predicting Recessions to Improve Asset Allocation Strategies


Overview
---
Conventional wisdom among investors often suggests that timing the stock market is a fool's errand. And while it is true that frequent trading in and out of the market rarely yields impressive results for the average investor, I would like to propose that a more disciplined, infrequent approach to timing can add value relative to a portfolio that holds a constant asset allocation over time. 

This proposal stems from an assumption that broader stock market cycles (Bull and Bear markets in Wall Street parlance) are tied to other, more predictable events, and by exploiting that predictability, investors can avoid the worst (though not all) of stock market declines. Specifically, it is not a revolutionary idea to suggest that stock market returns are stronlgy linked to economic cycles (expansions and recessions). As a result, if it can be shown that these economic cycles are to some degree predictable, it should logically follow that a portfolio management strategy can be developed to take advantage of that fact. 

In summary, this project aims to build a classification model that can predict the occurrence of economic recessions 12-months prior to their onset. Scores output by this model are then used to develop trading rules that aim to shift a theoretical investor's portfolio out of equities before recessions - and the stock market declines that typically accompany them - begin. The "stock market" referred to throughout this text is the S&P 500, and any money not invested in the market at any given time is assumed to be invested in 10-year U.S. Treasuries.  


Data
---
**Time Frame:** January 1982 - May 2020

The independent variables being used to train the model are the yield curve, expressed as the difference between yields on 10-yr and 3-month US Treasury securities, the Civilian Unemployment Rate (U3), and the NFCI Nonfinancial Leverage subindex. These variables are aggregated by month, and then smoothed using a 3-month SMA. The dependent variable is a recession indicator, with one (1) representing the occurrence of recession, and zero (0) the absence of recession. This indicator is shifted 12-months into the past as stock market declines do not perfectly coincide with the onset of recessions, and the aim here is to give investors sufficient advanced warning to lower their equity allocation. All data used to build the recession prediction model is obtained from the Federal Reserve Bank of St. Louis Economic Database (FRED), while S&P500 return data is drawn from Professor Robert Shiller's publically available dataset. 

**Sources:**
1. 10YR - 3M Treasury spread. (T10Y3M column in notebook, Frequency = Daily):  https://fred.stlouisfed.org/series/T10Y3M 
2. Recession Indicator (USREC, Frequency = Monthly): https://fred.stlouisfed.org/series/USREC 
3. Unemployment rate (UNRATE, Frequency = Monthly):  https://fred.stlouisfed.org/series/UNRATE
4. NFCI Nonfinancial Leverage Subindex (Frequency = Weekly): https://fred.stlouisfed.org/series/NFCINONFINLEVERAGE
5. 10-yr constant maturity rate (DGS10): https://fred.stlouisfed.org/series/DGS10
6. S&P 500 data: http://www.econ.yale.edu/~shiller/data.htm
7. Consumer Price Index (CPI): https://fred.stlouisfed.org/series/CPIAUCSL 


Methodology
---
**Train | Test Split**  
I divided the 1982-2020 period into training and test sets, with the dividing line between them being December, 2002. This provided a reasonable balance of both recessionary months and expansionary ones.

In selecting a model, I sought to choose a classifier and associated hyperparameters that maximized a given model's F_Beta score. Beta in this case was the ratio of the S&P 500's mean monthly returns during recessions to the same index's mean monthly returns during expansions. My aim in using this metric was to balance the goal of shifting out of the market before a recession hits with the desire to remain invested during the majority of the market’s uptrends. Out of the models I tested, **Logistic Regression** performed best in this regard.

After choosing a model I used the model's output scores to create rules that would shift an investor's portfolio out of the stock market sufficiently ahead of economic downturns. To do that I ran a number of simulations on a hypothetical portfolio, where each simulation adjusts the values of three variables to find the combination of values that yields the maximum portfolio return over the training time frame. These variables are:
1. The model score at which to lower the portfolio's equity allocation
2. The weights to shift the portfolio into once that model score is hit, and
3. The model score at which to resume the portfolio's baseline asset allocation 

For the second variable, the simulations were "instructed" to keep the portfolio's stock weight at or under the baseline. In other words, with a 60 | 40 portoflio the stock weight could go no higher than 60%. Other than when the rules counsel a shift in asset allocation, portfolio weights are kept constant, and the default test case for developing these rules was a classic 60 | 40 (stocks | bonds) portfolio.    



Results
---
Stated formally, the trading rules resulting in the highest portfolio returns were: Shift out of the stock market into a 10 | 90 (stocks | bonds) allocation once the model score hits 0.70, and hold that allocation until the model score reaches 0.10 or lower (at which point the portfolio returns to its baseline 60 | 40 allocation). Below is a plot showing these rules being carried out over the training set period (1982-2002).

![alt text](https://github.com/ErickDWalker/Recession_Prediction/blob/master/img/portfolio_weights_60:40_return_train.png?raw=true)  


Over the test set time frame (January, 2003 - May, 2020) a 60 | 40 portfolio that followed the above trading rules would have earned a 9.44% CAGR versus the passsive 60 | 40 portfolio's 7.77%. To put that into dollars, an initial investment of $10,000 in the passive strategy would have grown to $36,634, while the trading portfolio would have grown to $47,779 (plot of the growth of $10,000 in various portfolios shown below). In addition the trading portfolio achieved this outperformance with a lower standard deviation of returns (1.87% vs 2.08%).  


![alt text](https://github.com/ErickDWalker/Recession_Prediction/blob/master/img/60:40_test.png?raw=true)
