import pandas as pd
import pandas_datareader.data as web 
import datetime as dt 
import os 
import matplotlib.pyplot as plt 
import mplfinance as mpf
import yfinance as yf
import finplot as fplt

#loads and stores a stock price between two dates
def get_and_store_stock_price(stock='', source='yahoo', start=[2000, 1, 1], end=[2021, 1, 1], interval='d'):
	if len(start) < 3 or len(end) < 3:
		print("wrong date format")
		return -1
	
	Tstart = dt.datetime(start[0], start[1], start[2])
	Tend = dt.datetime(end[0], end[1], end[2])

	df = web.get_data_yahoo(stock, Tstart, Tend, interval=interval)
	if os.getcwd() == 'C:\\':
		os.chdir('C:\\Users\\mmart\\Documents\\Python\\finance\\backtester')

	df.to_csv(stock+interval+'.csv')

def plot_df(df, datesAth=None, datesLow=None, log=True):
	#mpf.plot(df, hlines=[100,1000], type='candle')

	if datesAth is not None and datesLow is not None:
		fig = mpf.figure()
		ax1 = fig.add_subplot(1,2,1,style='yahoo')
		ax2 = fig.add_subplot(1,2,2,style='yahoo')
		mpf.plot(df, ax=ax1, vlines=dict(vlines=datesAth), type='candle')
		mpf.plot(df, ax=ax2, vlines=dict(vlines=datesLow), type='candle')
		plt.show()
	else:
		if log:
			fig, axlist = mpf.plot(df, type='candle', style='yahoo', returnfig=True)
			ax = axlist[0]
			ax.set_yscale('log')
			plt.show()
		else:
			mpf.plot(df, type='candle', style='yahoo', returnfig=True)
			plt.show()

# find ath and X drawdown dates (original yahoo df)
def find_drawdown_and_ath(df, drawdown=50, start=None, end=None):
	
	if start is not None or end is not None:
		if start is None:
			start = df.iloc[0]['Date']
		if end is None:
			end = df.iloc[-1]['Date']
		
		new_df = get_time_period(df, start, end)
		print(df.head())
		print(new_df.head())
	else:
		new_df = df.copy()

	day_ath = []
	day_low = []
	price_ath = []
	price_low = []
	
	ath = 0
	drawdown_price = 0
	entry_found = True
	init_row = new_df.index[0]
	for row in new_df.index:
		# find new ath and set date 
		if new_df.iloc[row-init_row]['High'] > ath:
			if new_df.iloc[row-init_row]['High'] != 0:
				ath = new_df.iloc[row-init_row]['High']
				day = new_df.iloc[row-init_row]['Date']
				entry_found = False # keep track of ath in order to not assign day_low each month the price is down more than 50%

		# find day when price is 50% down
		elif new_df.iloc[row-init_row]['Low'] < ath * ((100-drawdown)/100) and entry_found == False:
			if new_df.iloc[row-init_row]['Low'] != 0:
				print("#"*10,"data","#"*10)
				print("ATH at %s, price: %.4f" % (day, ath))
				print("%d drawdown at %s, price low: %.4f" % (drawdown, new_df.iloc[row-init_row]['Date'], new_df.iloc[row-init_row]['Low']))
				print("#"*10,"data","#"*10)

				day_ath.append(day)
				price_ath.append(ath)
				
				day_low.append(new_df.iloc[row-init_row]['Date'])
				price_low.append(new_df.iloc[row-init_row]['Low'])
				entry_found = True
	
	day_ath.append(day)	

	print("Ath's:")
	for val in day_ath:
		print(val)

	print("%s drawdowns:" % drawdown)
	for val in day_low:
		print(val)

	return day_ath, day_low, price_ath, price_low

# find entry points 
# find ath and X drawdown dates (original yahoo df)
def find_entry_points(df, drawdown=50, show=False):
	day_ath = []
	day_low = []
	price_ath = []
	price_low = []
	
	ath_price = 0
	drawdown_price = 0

	drawdown_found = True
	entry_found = False

	for row in df.index:
		
		# find new ath and set date 
		if df.iloc[row]['High'] > ath:
			ath_price = val
			day = df.iloc[row]['Date']
			drawdown_found = False # keep track of ath in order to not assign day_low each month the price is down more than 50%

		# find day when price is 50% down
		if df.iloc[row]['Low'] < ath_price * ((100-drawdown)/100) and drawdown_found == False:
			drawdown_price = val
			day_ath.append(day)
			price_ath.append(ath)
			day_low.append(df.iloc[row]['Date'])
			price_low.append(df.iloc[row]['Low'])
			drawdown_found = True

	
	day_ath.append(day)	

	if show:
		print("Ath's:")
		for val in day_ath:
			print(val)

		print("%s drawdowns:" % drawdown)
		for val in day_low:
			print(val)

	return day_ath, day_low, price_ath, price_low

def convert_data(df):

	# convert dates
	conv_dates = []
	for date in df['Date']:
		if "Jan" in date:
			month = '01'
		if "Feb" in date:
			month = '02'
		if "Mar" in date:
			month = '03'
		if "Apr" in date:
			month = '04'
		if "May" in date:
			month = '05'
		if "Jun" in date:
			month = '06'
		if "Jul" in date:
			month = '07'
		if "Aug" in date:
			month = '08'
		if "Sep" in date:
			month = '09'
		if "Oct" in date:
			month = '10'
		if "Nov" in date:
			month = '11'
		if "Dec" in date:
			month = '12'

		year = '20' + date[date.find(" ")+1:]
		comp_date = year + '-' + month + '-01'
		conv_dates.append(comp_date)
	# convert prices (remove comma)
	conv_prices = []
	for row in df.index:
		Price = float(df.iloc[row]['Price'].replace(',', ''))
		Open = float(df.iloc[row]['Open'].replace(',', ''))
		High = float(df.iloc[row]['High'].replace(',', ''))
		Low = float(df.iloc[row]['Low'].replace(',', ''))
		conv_prices.append([Open, High, Low, Price])
	# convert volume 
	conv_vol = []
	for vol in df['Vol.']:
		if vol.find('K') != -1:
			conv_vol.append(round(float(vol.replace('K', ''))*1000, 2))
		elif vol.find('M') != -1:
			conv_vol.append(round(float(vol.replace('M', ''))*1000000, 2))
		else:
			print("Weird volume, check file....")
			return -1
	
	# put into new df 
	new_df = df.copy()

	# Set Price Column as Close, Vol. as Volume, Remove change
	new_df.drop("Change %", axis=1, inplace=True)
	new_df.drop("Vol.", axis=1, inplace=True)
	new_df.drop("Price", axis=1, inplace=True)

	# create some new columns and assign data
	new_df['Date'] = conv_dates
	new_df['Open'] = [item[0] for item in conv_prices]
	new_df['High'] = [item[1] for item in conv_prices]
	new_df['Low'] = [item[2] for item in conv_prices]
	new_df['Close'] = [item[3] for item in conv_prices]
	new_df['Volume'] = conv_vol

	new_df.set_index(new_df['Date'], inplace=True)
	new_df.drop("Date", axis=1, inplace=True)

	return new_df

# get date interval from dataframe
def get_time_period(dataframe, start, end, index=False):
	# greater than start date and smaller than the end date 
	start_date = start
	end_date = end
	mask = (dataframe['Date'] > start_date) & (dataframe['Date'] <= end_date)
	new_df = dataframe.loc[mask]
	if index:
		return new_df.set_index('Date')
	else:
		return new_df
# pick range from dataframe and plot with moving averages
def plot_price_in_range(df, name, start, end):
	new_df = get_time_period(df, start, end)
	ax = fplt.create_plot(name, rows=1)

	# plot candle stick
	candles = new_df[['Open','Close','High','Low']]
	fplt.candlestick_ochl(candles, ax=ax)

	# moving averages 
	fplt.plot(new_df.Close.rolling(50).mean(), legend='ma50')
	fplt.plot(new_df.Close.rolling(100).mean(), legend='ma100')
	fplt.plot(new_df.Close.rolling(150).mean(), legend='ma150')
	fplt.plot(new_df.Close.rolling(200).mean(), legend='ma200')

	# overlay volume on the top plot
	volumes = new_df[['Open', 'Close', 'Volume']]
	fplt.volume_ocv(volumes, ax=ax.overlay())

	fplt.show()

# backtest dB
def backtest_dB(df, datesLow, start_money=1000, monthly_savings=100, buy_initial=False):

	print("#"*10, 'Backtest "dB"', "#"*10)
	print("Starting balance: %d USD" % start_money)
	print("Monthly savings: %d USD" % monthly_savings)
	
	monthly_total = []
	monthly_bitcoins = []
	monthly_cash = []
	balance = start_money
	numBtc = 0
	idx = 1
	for row in df.index:
		balance = balance + monthly_savings

		if row == 0 and buy_initial:
			numBtc = numBtc + (balance / df.iloc[row]['Close'])
			balance = 0
		# if drawdown, go all in
		if df.iloc[row]['Date'] in datesLow:
			numBtc = numBtc + (balance / df.iloc[row]['Close'])
			#print("%d Drawdown: Number of Bitcoins accumuated by %s: %f (Price: %.2f)" % (idx, df.iloc[row]['Date'], numBtc, df.iloc[row]['Close']))
			balance = 0
			idx = idx + 1

		monthly_cash.append(balance)
		monthly_bitcoins.append(numBtc)
		monthly_total.append(balance + numBtc*df.iloc[row]['Close'])


	btcWorth = numBtc*df.iloc[len(df)-1]['Open']
	cagr = (pow(((btcWorth+balance)/(start_money+monthly_savings)),1/int(len(df)/12))-1) * 100

	print("#"*10, 'Result %s' % df.iloc[len(df)-1]['Date'], "#"*10)	
	print("Final number of bitcoins accumulated: ", numBtc)
	print("Current value of Bitcoins: %.2f USD " % (btcWorth))
	print("Current Account balance: %d USD" % balance)
	print("Capital: %.4f USD" % (balance + btcWorth))
	print("CAGR: %.2f %%" % cagr)

	return monthly_total, monthly_cash, monthly_bitcoins

# dc monthly 
def dc_monthly(df, start_money=1000, monthly_savings=100):
	print("#"*10, 'Buy every Monthly Close', "#"*10)
	print("Starting balance: %d USD" % start_money)
	print("Monthly savings: %d USD" % monthly_savings)
	
	monthly_total = []
	monthly_bitcoins = []
	monthly_cash = []
	balance = start_money
	numBtc = 0
	for row in df.index:
		balance = balance + monthly_savings
		numBtc = numBtc + (balance / df.iloc[row]['Close'])
		balance = 0

		monthly_cash.append(balance)
		monthly_bitcoins.append(numBtc)
		monthly_total.append(balance + numBtc*df.iloc[row]['Close'])

	btcWorth = numBtc*df.iloc[len(df)-1]['Open']
	cagr = (pow(((btcWorth+balance)/(start_money+monthly_savings)), 1/int(len(df)/12))-1) * 100

	print("#"*10, 'Result %s' % df.iloc[len(df)-1]['Date'], "#"*10)	
	print("Final number of bitcoins accumulated: ", numBtc)
	print("Current value of Bitcoins: %.2f USD " % (btcWorth))
	print("Current Account balance: %d USD" % balance)
	print("Capital: %.4f USD" % (balance + btcWorth))
	print("CAGR: %.2f %%" % cagr)

	return monthly_total, monthly_cash, monthly_bitcoins

# backtest buy every ath before drawdown
def buy_ath_before_drawdown(df, datesHigh, start_money=1000, monthly_savings=100):

	print("#"*10, 'Backtest "dB"', "#"*10)
	print("Starting balance: %d USD" % start_money)
	print("Monthly savings: %d USD" % monthly_savings)
	
	monthly_total = []
	monthly_bitcoins = []
	monthly_cash = []
	balance = start_money
	numBtc = 0
	idx = 1
	for row in df.index:
		balance = balance + monthly_savings

		# if drawdown, go all in
		if df.iloc[row]['Date'] in datesHigh:
			numBtc = numBtc + (balance / df.iloc[row]['High'])
			print("%d Ath: Number of Bitcoins accumuated by %s: %f (Price: %.2f)" % (idx, df.iloc[row]['Date'], numBtc, df.iloc[row]['High']))
			balance = 0
			idx = idx + 1

		monthly_cash.append(balance)
		monthly_bitcoins.append(numBtc)
		monthly_total.append(balance + numBtc*df.iloc[row]['Close'])

	btcWorth = numBtc*df.iloc[len(df)-1]['Open']
	cagr = (pow(((btcWorth+balance)/(start_money+monthly_savings)),1/int(len(df)/12))-1) * 100

	print("#"*10, 'Result %s' % df.iloc[len(df)-1]['Date'], "#"*10)	
	print("Final number of bitcoins accumulated: ", numBtc)
	print("Current value of Bitcoins: %.2f USD " % (btcWorth))
	print("Current Account balance: %d USD" % balance)
	print("Capital: %.4f USD" % (balance + btcWorth))
	print("CAGR: %.2f %%" % cagr)

	return monthly_total, monthly_cash, monthly_bitcoins

# plot result from strategies above 
def plot_result(df, total, mc, mbtc):
	total = [val/1000000 for val in total]
	month = [index for index in df.index]

	fig, ax = plt.subplots(2,2)
	ax[0,1].plot(month, total)
	ax[1,1].plot(month, mc)
	ax[1,0].plot(month, mbtc)
	
	ax[0,1].set_ylabel("MUSD")
	ax[0,1].set_xlabel("month")
	ax[0,1].set_title("Balance / Months")
	
	ax[1,1].set_ylabel("Cash USD")
	ax[1,1].set_xlabel("month")
	ax[1,1].set_title("Cash / Months")

	ax[1,0].set_ylabel("Number of bitcoins")
	ax[1,0].set_xlabel("month")
	ax[1,0].set_title("Bitcoins / Months")

	plt.tight_layout()
	plt.show()

def cm_to_inch(value):
    return value/2.54

def make_gif(df, name, fsize=[3, 3]):
	os.chdir('C:\\Users\\mmart\\Documents\\Python\\finance\\backtester')
	name = name.replace(".", "_")
	monthly = False
	if name.find("m_") != -1:
		monthly = True

	if not os.path.exists(name):
		os.makedirs(name)
	os.chdir('C:\\Users\\mmart\\Documents\\Python\\finance\\backtester\\'+name)

	filenames = []
	for row in df.index:
		# print last frame multiple times to give a "freeeze"
		if row == len(df) - 1:
			# print frame 20 times 
			for i in range(20):
				plt.figure(figsize=(fsize[0], fsize[1]))
				# plot 
				plt.plot(df.iloc[:row]['Close'])
				# y lim
				plt.ylim(0,int(df.iloc[row]['Close']*2))

				plt.title(name+" Price Chart")

				if monthly:
					plt.ylabel("Price Monthly Close (SEK)")
					plt.xlabel("Months since IPO")
				else:
					plt.ylabel("Price Daily Close (SEK)")
					plt.xlabel("Days since IPO")

				# create file name and append it to a list
				filename = f'{row+i}.png'
				filenames.append(filename)
				
				# save frame
				plt.savefig(filename)
				plt.close()
		else:
			# plot 
			plt.plot(df.iloc[:row]['Close'])
			# y lim
			plt.ylim(0,int(df.iloc[row]['Close']*2))

			plt.title(name+" Price Chart")

			if monthly:
				plt.ylabel("Price Monthly Close (SEK)")
				plt.xlabel("Months since IPO")
			else:
				plt.ylabel("Price Daily Close (SEK)")
				plt.xlabel("Days since IPO")
			
			# create file name and append it to a list
			filename = f'{row}.png'
			filenames.append(filename)
			
			# save frame
			plt.savefig(filename)
			plt.close()
	
	# build gif
	with imageio.get_writer(name+str(fsize[0]).replace(".","_")+str(fsize[1]).replace(".","_")+'.gif', mode='I') as writer:
		for filename in filenames:
			image = imageio.imread(filename)
			writer.append_data(image)

	# Remove files
	for filename in set(filenames):
		os.remove(filename)

# ------------------------------------------------------------------ BEGIN get and store
# get csv file of a stock 
#stock = 'EVO.ST'
#source = 'yahoo'
#start = [2010,1,1]
#end = [2021,4,4]

#get_and_store_stock_price(stock, source, start, end, interval='d')
# ------------------------------------------------------------------ END get and store

# ------------------------------------------------------------------ BEGIN visualize
# if program started from c:\ in git bash
if os.getcwd() == 'C:\\':
	os.chdir('C:\\Users\\mmart\\Documents\\Python\\finance\\backtester') 

# CONVERT ####################################################
# convert format of dates etc when data comes from investing.com downloaded csv 
#stock = 'BTC-USDmch.csv'
#df = pd.read_csv(stock)
#new_df = convert_data(df)

#new_df.to_csv(stock.replace('.csv', '')+'Mod.csv')

# END CONVERT ##############################################
# plot 
#stock = 'BTC-USDmchMod.csv'
#df = pd.read_csv(stock)
#start = '2010-01-01'
#end = '2017-03-21'
#plot_df(df) # expect Date as index
#plot_price_in_range(df, stock, start, end) # expects index 

# MAKE GIF ###################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

stock = 'EVO.STd.csv'
df = pd.read_csv(stock)
make_gif(df, stock, fsize=[0.01,0.01])

#y = np.random.randint(30, 40, size=(40))

## ONE ##
'''
plt.plot(y[:-3])
plt.ylim(20,50)
plt.savefig('1.png')
plt.show()
## TWO ##
plt.plot(y[:-2])
plt.ylim(20,50)
plt.savefig('2.png')
plt.show()
## THREE ##
plt.plot(y[:-1])
plt.ylim(20,50)
plt.savefig('3.png')
plt.show()
## FOUR ##
plt.plot(y)
plt.ylim(20,50)
plt.savefig('4.png')
plt.show()
'''

# END MAKE GIF ###################################################





# find buy zone and plot dates 
# BEGIN DB MODELLEN ########################################################################
#stock = 'BTC-USDmchMod.csv'
#df = pd.read_csv(stock)
#start = '2015-01-01'

#datesAth, datesLow, pAth, pLow = find_drawdown_and_ath(df)
#total, mc, mbtc = backtest_dB(df, datesLow, start_money=1000, monthly_savings=100)
#total, mc, mbtc = backtest_dB(df, datesLow, start_money=0, monthly_savings=100)
#total, mc, mbtc = backtest_dB(df, datesLow, start_money=0, monthly_savings=100, buy_initial=True)
#total, mc, mbtc = dc_monthly(df, start_money=0, monthly_savings=100)
#plot_result(df, total, mc, mbtc)



#datesAth, datesLow, pAth, pLow = find_drawdown_and_ath(df, start=start)
#total, mc, mbtc = backtest_dB(df, datesLow, start_money=1000, monthly_savings=100)
#plot_result(df, total, mc, mbtc)
# END DB MODELLEN ########################################################################


# ------------------------------------------------------------------ END visualize

# ------------------------------------------------------------------ BEGIN TO DO 
# 1. build ma strategies based on
# * any stock
# * interval
# * ma X / Y buy / sell rules
# * compare to dca each month 
# 2. Build trend lines model 
# * find three following ath
# * find three following lows
# * plot trend channel
# * dca when in trend
# * sell when price < line

# ------------------------------------------------------------------ END TO DO