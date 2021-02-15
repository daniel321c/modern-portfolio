import csv

daily_prices = list(csv.DictReader(open("/Users/daniel/Downloads/AAPL.csv")))


continued_inc = 0
continued_dec = 0 
for price in daily_prices:
    try:
        price['increased'] = float(price['Close']) - float(price['Open'])  > 0
    except:
        price['increased'] = False
        print('error')

    price['continued_dec'] = continued_dec
    price['continued_inc'] = continued_inc
    
    if price['increased']:
        continued_inc += 1
        continued_dec = 0
    else:
        continued_inc = 0
        continued_dec +=1

# stock increase or decrease after X continued increase
distr = {}
for price in daily_prices:
    if price['increased']:
        if price['continued_inc'] not in distr:
            distr[price['continued_inc']] = {'up': 1, 'down': 0}
        else:
            distr[price['continued_inc']]['up'] +=1
    else:
        if price['continued_inc'] not in distr:
            distr[price['continued_inc']] = {'up': 0, 'down': 1}
        else:
            distr[price['continued_inc']]['down'] +=1
print(distr)



# stock increase or decrease after X continued decrease
distr = {}

for price in daily_prices:
    if price['increased']:
        if price['continued_dec'] not in distr:
            distr[price['continued_dec']] = {'up': 1, 'down': 0}
        else:
            distr[price['continued_dec']]['up'] +=1
    else:
        if price['continued_dec'] not in distr:
            distr[price['continued_dec']] = {'up': 0, 'down': 1}
        else:
            distr[price['continued_dec']]['down'] +=1
print('\n')
print(distr)
