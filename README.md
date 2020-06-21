# Predicting Recessions to Improve Asset Allocation Strategies


Overview
---
Conventional wisdom among investors often suggests that timing the stock market is a fool's errand. And while it is true that rapid trading in and out of the market rarely yields impressive results for the average investor, I would like to propose that a more disciplined, infrequent approach to market timing can add value relative to a portfolio that holds a constant asset allocation over time. 

This proposal stems from an assumption that broader stock market cycles (Bull and Bear markets in Wall Street parlance) are tied to other, more predictable events, and by exploiting that predictability, investors can avoid the worst (though not all) of stock market declines. Specifically, it is not a revolutionary idea to suggest that stock market returns are strongly linked to economic cycles (see plot below). As a result, if it can be shown that these economic cycles are to some degree predictable, it should logically follow that a portfolio management strategy can be developed to take advantage of that fact. 

![alt text](https://github.com/ErickDWalker/Recession_Prediction/blob/master/img/S&P500_returns.png?raw=true)  

In summary, this project aims to build a classification model that can predict the occurrence of economic recessions 12-months prior to their onset. Scores output by this model are then used to develop trading rules that aim to shift a theoretical investor's portfolio out of equities before recessions - and the often substantial stock market declines that accompany them - begin. The "stock market" referred to throughout this text is the S&P 500, and any money not invested in the market at any given time is assumed to be invested in 10-year U.S. Treasuries.  


Data
---
**Time Frame:** January 1982 - May 2020

The independent variables being used to train the model are the yield curve, expressed as the difference between yields on 10-yr and 3-month US Treasury securities, the Civilian Unemployment Rate (U3), and the NFCI Nonfinancial Leverage subindex. These variables are aggregated by month, and then smoothed using a 3-month SMA. The dependent variable is a binary indicator, with one (1) representing the occurrence of recession as defined by the National Bureau of Economic Research, and zero (0) the absence of one. To account for the fact that stock market declines do not perfectly coincide with the onset of recessions, this indicator is shifted 12-months into the past, with the aim of providing investors sufficient warning in advance to lower their equity exposure. All data used to build the recession prediction model is obtained from the Federal Reserve Bank of St. Louis Economic Database (FRED), while S&P500 return data is drawn from Professor Robert Shiller's publically available data set. 

**Sources:**
1. 10YR - 3M Treasury spread. (T10Y3M column in notebook, Frequency = Daily):  https://fred.stlouisfed.org/series/T10Y3M 
2. Recession Indicator (USREC, Frequency = Monthly): https://fred.stlouisfed.org/series/USREC 
3. Unemployment rate (UNRATE, Frequency = Monthly):  https://fred.stlouisfed.org/series/UNRATE
4. NFCI Nonfinancial Leverage Subindex (Frequency = Weekly): https://fred.stlouisfed.org/series/NFCINONFINLEVERAGE
5. 10-yr constant maturity rate (DGS10): https://fred.stlouisfed.org/series/DGS10
6. S&P 500 data: http://www.econ.yale.edu/~shiller/data/ie_data.xls
7. Consumer Price Index (CPI): https://fred.stlouisfed.org/series/CPIAUCSL 


Methodology
---
**Train | Test Split**  
I split the 1982-2020 period into training and test sets, with the dividing line between them being December, 2002. This provided a reasonable balance of recessionary and expansionary months in both sets.

In selecting a model, I sought to choose a classifier and associated hyperparameters that produced the maximum F_Beta score. My aim in using this metric was to allow for a consideration of both recall and precision, thereby balancing the goal of shifting out of the market before a recession hits with the desire to remain invested during the majority of the market’s uptrends. Beta in this case was the ratio of the S&P 500's mean monthly returns during recessions to the same index's mean monthly returns during expansions. Out of the models I tested, **Logistic Regression** performed best in this regard. Below is a plot of the scores produced when applying the model's *predict* method to the entire data set (after fitting the model on the training data set).

![alt text](https://github.com/ErickDWalker/Recession_Prediction/blob/master/img/Logistic_Regression_Output.png?raw=true)

After choosing a model I used the model's output scores (the maroon line in the plot above) to create rules that would shift an investor's portfolio out of the stock market sufficiently ahead of economic downturns. To do that I ran a number of simulations on a hypothetical portfolio, where each simulation adjusts the values of three variables to find the combination of values that yields the maximum portfolio return over the training time frame. These variables are:
1. The model score at which to lower the portfolio's equity allocation
2. The weights to shift the portfolio into once that model score is hit, and
3. The model score at which to resume the portfolio's baseline asset allocation 

Other than when the rules counsel a shift in asset allocation, portfolio weights are kept constant at a baseline allocation that can be chosen in advance by the user. The default baseline case used in the notebook for developing the rules was a classic 60 | 40 (stocks | bonds) portfolio. In deciding what asset weights to shift into (variable #2), the simulations were "instructed" to keep the portfolio's stock weight at or under the baseline's. In other words, with the default baseline 60 | 40 portfolio the stock weight was not allowed to be set higher than 60%. 


Results
---
Stated formally, the trading rules resulting in the highest portfolio returns over the training set were: Shift out of the stock market into a 10 | 90 (stocks | bonds) allocation once the model score hits 0.70, and hold that allocation until the model score reaches 0.10 or lower (at which point the portfolio returns to its baseline 60 | 40 allocation). Below is a plot showing these rules being carried out over the training set period (1982-2002).

![alt text](https://github.com/ErickDWalker/Recession_Prediction/blob/master/img/portfolio_weights_60:40_return_train.png?raw=true)

Over the test set time frame (January, 2003 - May, 2020) a 60 | 40 portfolio that followed the above trading rules would have adjusted its asset allocations as shown in the plot below.

![alt text](https://github.com/ErickDWalker/Recession_Prediction/blob/master/img/portfolio_weights_60:40_return_test.png?raw=true)

It bears mentioning that while the rule successfully reduced the portfolio's equity allocation during the 2008-2009 recession,  it failed to do so during the recession started March 2020 (the model's score didn't reach the rule's 0.70 level). Though this is far from ideal, I think two things are worth noting. First, the fact that the model's score failed to reach the 0.70 level triggering an allocation shift is not all that surprising considering the nature of the current recession. The cause of the decline in GDP in the U.S., and the speed at which it happened, are unprecedented in not only the time frame over which the model was trained, but all of U.S. history. This would understandably make it more difficult for the model to make accurate predictions. Second, even with this "miss", the activation of the trading rules during the 2008-2009 recession was enough to lead to meaningfully superior gains for a portfolio over the Jan 2003 - Mar 2020 window. 

The 60| 40 portfolio that adjusted asset weights according to the trading rules, earned a 9.44% CAGR versus the passsive 60 | 40 portfolio's 7.77%. Translating that outperformance into dollars, an initial investment of $10,000 in the passive strategy would have grown to $36,634, while the trading portfolio would have grown to $47,779 (plot of the growth of $10,000 in various portfolios shown below). In addition the trading portfolio achieved this outperformance with a lower standard deviation of returns (1.87% vs 2.08%).  


![alt text](https://github.com/ErickDWalker/Recession_Prediction/blob/master/img/60:40_test.png?raw=true)
