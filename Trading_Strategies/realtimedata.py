from ib_async import * 
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timezone 
from IPython.display import display, clear_output
import os # new
import matplotlib.pyplot as plt

# ib = IB()
# if ib.isConnected():
#     print("Connected to IBKR API")
# else: 
#     ib.connect()
#     print(f"Connected to IBKR API: {ib.isConnected()}")

# # strategy parameters
# contract = Crypto('BTC', 'PAXOS', 'USD')  # or 'ZEROHASH' depending on your IBKR permissions
# ib.qualifyContracts(contract)
# ticker = ib.reqMktData(contract)
# ib.sleep(1)  # give it time to populate

# def onPendingTickers(tickers): # what shall happen after receiving a new tick
#     global message
#     message = "time: {} | Bid: {} | Ask:{}".format(ticker.time, ticker.bid, ticker.ask)
#     print(message, end = '\r')

# ib.pendingTickersEvent += onPendingTickers # activate onPendingTickers
# ib.sleep(30) # new, to be added!!!
# ib.pendingTickersEvent -= onPendingTickers # de-activate onPendingTickers
# ib.cancelMktData(contract) 

# #####  MULTIPLE TICKER DATA ########

# contracts = [
#     Crypto('BTC', 'PAXOS', 'USD'),
#     Crypto('ETH', 'PAXOS', 'USD'),
#     Crypto('SOL', 'PAXOS', 'USD'),
# ]  # or use 'ZEROHASH' depending on your IBKR permissions
# ib.qualifyContracts(*contracts)

# def contract_key(contract):
#     return f"{contract.symbol}-{contract.exchange}"

# for contract in contracts:
#     ib.reqMktData(contract)

# df = pd.DataFrame(
# index=[contract_key(c) for c in contracts],
# columns=['bidSize', 'bid', 'ask', 'askSize', 'high', 'low', 'close'])
# print(df) 

# def onPendingTickers(tickers): # what shall happen after receiving a new tick
#     for t in tickers:
#         df.loc[contract_key(t.contract)] = ( # pyright: ignore[reportCallIssue] # type: ignore
#             t.bidSize, t.bid, t.ask, t.askSize, t.high, t.low, t.close)
#     clear_output(wait=True)
#     display(df)

# ib.pendingTickersEvent += onPendingTickers # activate onPendingTickers
# ib.sleep(30) # new to be added!!!
# ib.pendingTickersEvent -= onPendingTickers # de-activate onPendingTickers

# for contract in contracts:
#     ib.cancelMktData(contract)

# ############### STREAMING BAR DATA, Historical and live ################
# contract = Crypto('BTC', 'PAXOS', 'USD')

# def onBarUpdate(bars, hasNewBar):  # what shall happen after receiving a new bar
#     global df
#     df = pd.DataFrame(bars)[["date", "open", "high", "low", "close"]]
#     df.set_index("date", inplace = True)
#     clear_output(wait=True)
#     display(df)

# # start stream
# bars = ib.reqHistoricalData(
#         contract,
#         endDateTime='',
#         durationStr='50 S',
#         barSizeSetting='5 secs',
#         whatToShow='MIDPOINT',
#         useRTH=True,
#         formatDate=2,
#         keepUpToDate=True)

# bars.updateEvent += onBarUpdate # activate onBarUpdate
# ib.sleep(30) # new to be added!!!
# bars.updateEvent -= onBarUpdate # de-activate onBarUpdate
# ib.cancelHistoricalData(bars) # stop stream 

############# Creating a live candle stick chart ################
# contract = Crypto('BTC', 'PAXOS', 'USD')

# # Keep one interactive figure alive and redraw into it.
# plt.ion()
# fig, ax = plt.subplots()

# def onBarUpdate(bars, hasNewBar):
#     global fig, ax
#     ax.clear()
#     plt.sca(ax)
#     util.barplot(bars, title="BTC", upColor="green", downColor="red")
#     fig.canvas.draw_idle()
#     fig.canvas.flush_events()
#     plt.pause(0.01)

# bars = ib.reqHistoricalData(
#         contract,
#         endDateTime='',
#         durationStr='1000 S',
#         barSizeSetting='10 secs',
#         whatToShow='MIDPOINT',
#         useRTH=True,
#         formatDate=2,
#         keepUpToDate=True)

# bars.updateEvent += onBarUpdate

# try:
#     ib.sleep(30)
# except KeyboardInterrupt:
#     print("Interrupted by user. Shutting down stream...")
# finally:
#     try:
#         bars.updateEvent -= onBarUpdate
#     except ValueError:
#         pass
#     ib.cancelHistoricalData(bars)
#     plt.ioff()
#     plt.close(fig)
#     if ib.isConnected():
#         ib.disconnect()

############# Preparing data for day trading ##########################

