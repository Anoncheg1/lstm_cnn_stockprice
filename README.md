# pytorch_lstm_cnn_stockprice
Stock price forecasting for MOEX

Tensorflow project. First Pytorch version was not success.

The idea is to use LSTM for time series prediction and CNN for
recognition of visual patterns of image which see broker on the
screen.
Savitzky-Golay used to overcame overfitting by controling smoothing of the date.

![Alt text](https://github.com/Anoncheg1/lstm_cnn_stockprice/blob/main/tfmy/MGNT_150130_200204.csv.jpg)

### files
augum/da305.py - used data: close, volume, time, open, high, low
augum/da_cnn_lstm.py - close, volume
tfmy/cnn_lstm.py - train and predict steps

### links
CNN vs. Prophet: Forecasting the Copper Producer Price Index https://towardsdatascience.com/cnn-vs-prophet-forecasting-the-copper-producer-price-index-af4da63bd93d

Channel and Spatial Attention CNN: Predicting Price Trends from Images https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4173579

Time Series Data Prediction Using Sliding Window Based RBF Neural Network https://www.ripublication.com/ijcir17/ijcirv13n5_46.pdf

Savitzky-Golay filter for stocks and time series (savgol fit) with Python https://tcoil.info/savitzky-golay-filter-for-stocks-and-time-series-savgol-fit/
