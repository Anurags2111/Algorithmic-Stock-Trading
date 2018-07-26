strength = 50
counter_range = 20
nbepochs = 2
batch_size_ = 100
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Activation



#json file contents
filter_cols = ["Open", "Close", "Volume_(BTC)", "Volume_(Currency)"]
x_window_size = 150
y_window_size = 1
y_column = "Close"
df = pd.DataFrame(columns = ['Timestamp', 'Price', 'Return', 'Action'])
timestamp = []
cur_time = []
btcp = []
btcp_pred = []
xbuy = []
xsell = []
bp = []
sp = []
YY = []
Return = []
action = []
tradeprice = []
close = []
plane = []



dataset = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv')
dataset = dataset[::30]
dataset = dataset.reset_index(drop = True)




for counter in range(counter_range):
    
    up = counter * 1000 + 1999 + 149 + 2
    data = dataset[:up]
    
    time = data['Timestamp'][-1000:]
    timestamp = np.concatenate((timestamp, time), axis = 0)
    timestamp = timestamp.astype(int)
    
    if (filter_cols):
        # Remove any columns from data that we don't need by getting the difference between cols and filter list
        rm_cols = set(data.columns) - set(filter_cols)
        for col in rm_cols:
            del data[col]
    
    # Convert y-predict column name to numerical index
    y_col = list(data.columns).index(y_column)
    
    x_train = data[:-1000]
    x_test = data[-1000:]
    y_test = data['Close'][-1000:]
    
    #present_price = x_train['Close'][len(x_train)-1]
    
    x_col = x_train.columns
    
    sc_X = MinMaxScaler(feature_range=(0, 1))
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    
    #present_price_scaled = x_t['Close'][len(x_train)-1]
    
    x_train = pd.DataFrame(data=x_train, columns=x_col)
    x_test = pd.DataFrame(data=x_test, columns=x_col)
    y_test_scaled = x_test['Close'][-1000:]
    
    num_rows = len(x_train)
    x_data = []
    y_data = []
    i = 0
    while ((i + x_window_size + y_window_size) <= num_rows):
        x_window_data = x_train[i:(i + x_window_size)]
        y_window_data = x_train[(i + x_window_size):(i + x_window_size + y_window_size)]
        
        y_average = np.average(y_window_data.values[:, y_col])
        x_data.append(x_window_data.values)
        y_data.append(y_average)
        i += 1
        
    x_np_arr = np.array(x_data)
    y_np_arr = np.array(y_data)
    
    model = Sequential()
    model.add(LSTM(input_dim = x_np_arr.shape[2], output_dim=x_np_arr.shape[1], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(x_np_arr.shape[1], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim = 1))
    model.add(Activation("tanh"))
    model.compile(loss = "mse", optimizer = "Nadam")
    model.fit(x_np_arr, y_np_arr, epochs = nbepochs, batch_size = batch_size_)
    
    dataset_total = pd.concat((x_train, x_test), axis = 0)
    test = dataset_total[-1000-x_window_size+1:]
    
    num_rows = len(test)
    x_data = []
    i = 0
    while ((i + x_window_size) <= num_rows):
        test_window = test[i:(i + x_window_size)]
        x_data.append(test_window.values)
        i += 1
        
    x_test_arr = np.array(x_data)
    
    predicted_price = model.predict(x_test_arr).reshape(-1)
    predicted_price_df = pd.DataFrame(data = predicted_price, columns = ['pred'])
    y_test_scaled_df = pd.DataFrame(y_test_scaled)
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
  
    ret_pred = (predicted_price_df['pred'] / x_test['Close'] - 1) * 10000
    
    plt.plot(ret_pred)
    plt.show()

    buy = 0
    for var in range(len(ret_pred)):
        
        timeline = counter * 1000 + var
        
        if ret_pred[var] > 10 and buy == 0: # BUY
            Return.append(ret_pred[var])
            action.append('Bought')
            xbuy.append(timeline)
            plane.append(timeline)
            bp.append(btcp[var])
            tradeprice.append(btcp[var])
            cur_time.append(timestamp[var])
            buy = 1
        
        elif ret_pred[var] < -10 and buy == 1: # SELL
            Return.append(ret_pred[var])
            action.append('Sold')
            xsell.append(timeline)
            plane.append(timeline)
            sp.append(btcp[var])
            tradeprice.append(btcp[var])
            cur_time.append(timestamp[var])
            buy = 0

    YY = np.concatenate((YY, btcp), axis = 0)


ax0 = plt.plot(YY, color = 'orange', linewidth = 0.8)
ax1 = plt.scatter(xbuy, bp, color='g') #buy
ax2 = plt.scatter(xsell, sp, color='r') #sell
plt.legend((ax0, ax1, ax2), ('LTP', 'Bought', 'Sold'))

df['Timestamp'] = cur_time
df['Price'] =  tradeprice
df['Return'] =  Return
df['Action'] = action




#====================================** TRADE SHEET **======================================================================


TS = pd.DataFrame(columns = ['Timestamp', 'Price','Action','Position','TradeQ',
                             'Cash','BTC_Amt','Cu_Credit','Cu_Debit',
                             'Trade','Turnover','PNL','PNLperTrade'])

pos = []
Action = []
PNL = []
Price = []
tradq = []
trade = []
turn = []
Cash = []
btcamt = []
CC = []
CD = []
p_ = 0
pb_ = 0
ps_ = 0
r = 0
cur_pro = 0
mon = 0
btc = 0
credit = 0
debit =0
cumcredit = 0
cumdebit = 0
q1 = 0
q2 = 0
cur_trade = 0
pre_trade = 0
prev = 0
pnl = 0
Q = 0
time_ = []


for var in range(len(df)):

    btcp = df['Price'][var]
    # Buying
    if df['Action'][var] == 'Bought':
        time_.append(df['Timestamp'][var])
        Action.append('Bought')
        Price.append(btcp)
        p_ = 1
        btc = 1
        pos.append(p_)
        if r == 0:
            q2 = 1
            r = 1
        else:
            q2 = p_
        
        Q = q2-q1
        tradq.append(Q)
        q1 = q2
        mon -= btcp*Q
        Cash.append(mon)
        btcamt.append(p_*btcp)
        debit = (btcp*Q)
        cumdebit += debit
        CD.append(cumdebit)
        CC.append(cumcredit)
        trade.append(credit+debit)
        turn.append(cumcredit+cumdebit)
        pnl = cumcredit - cumdebit + p_*btcp
        PNL.append(pnl)

    # Selling
    if df['Action'][var] == 'Sold':
        time_.append(df['Timestamp'][var])
        Action.append('Sold')
        Price.append(btcp)
        p_ = -1
        btc = 0
        pos.append(p_)
        q2 = p_        
        Q = q2 - q1
        tradq.append(Q)
        q1 = q2
        mon += btcp*(-Q)
        Cash.append(mon)
        btcamt.append(p_*btcp)
        credit = (btcp*(-Q))
        cumcredit += credit
        CC.append(cumcredit)
        CD.append(cumdebit)
        trade.append(credit+debit)
        turn.append(cumcredit+cumdebit)
        pnl = cumcredit - cumdebit + p_*btcp
        PNL.append(pnl)

TS['Timestamp'] = time_
TS['Price'] = Price
TS['Action'] = Action
TS['Position'] = pos

TS['Cash'] = Cash
TS['BTC_Amt'] = btcamt
tradq = pd.Series(tradq)
TS['TradeQ'] = tradq
TS['Cu_Credit'] = CC
TS['Cu_Debit'] = CD

TS['Trade'] = trade
TS['PNL'] = PNL
TS['Turnover'] = turn

TS['PNLperTrade'] = TS['PNL'] / TS['Turnover']
TS['trans_cost'] = TS['Turnover'] * (0.1/100)
TS['PnL_after_TC'] = TS['PNL'] - TS['trans_cost']
#TS.to_csv('tradesheet.csv', index=False)


plt.figure(figsize = (20, 10))
plt.plot(TS['PnL_after_TC'])
plt.plot(TS['PNL'])
plt.legend()
plt.savefig('PNL.png')
plt.show()

