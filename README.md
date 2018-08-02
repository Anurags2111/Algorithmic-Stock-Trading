# Bitcoin Algorithmic Trading
## Description
The project task was to identify the favourable prices in future for trading (buying or selling) the bitcoin(s) for US Dollars. This would help figure out the future direction of the market price with respect to the current as well as the previous positions of the market prices.

## Dataset
The dataset is called 'coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv' and was taken from Kaggle Bitcoin Historical Data. It is a 4-years data containing 1819074 observations dated from 2014-12-01 to 2018-06-27. It was sampled in a time period of 30 minutes.

## Model
Model used was RNN-LSTM model containing 2 LSTM cells followed by a Dense layer. It takes input in batches of 100 observations and was run for 2 epochs.


![lsttm](https://user-images.githubusercontent.com/23147497/43596020-bdd3b042-969b-11e8-9f7a-710f05cfe6c1.png)


![forget](https://user-images.githubusercontent.com/23147497/43596022-bf391012-969b-11e8-91f8-08e7dcfcffbe.png)

![input](https://user-images.githubusercontent.com/23147497/43596024-c021e76a-969b-11e8-92f2-125bd807ae48.png)


![lstm3-focus-c](https://user-images.githubusercontent.com/23147497/43596028-c17ad8ba-969b-11e8-9a82-89e3b1859959.png)


![lstm3-focus-o](https://user-images.githubusercontent.com/23147497/43596029-c2837d0c-969b-11e8-800b-fb7cfae157fa.png)




## Input
Model input contains a 3D tensor containing 150 previous Closing prices and current values for Open, Close, Volume(BTC) and Volume(Currency).

## Output
Model predicts the next Closing value (after 30 minutes).
