import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from itertools import chain, combinations

from flask import Flask, render_template,jsonify,request,abort,Response
from datetime import datetime
import json
global rules
data = pd.read_csv('groceryinfo.csv', header = None)
#print(data)
records = []
for i in range(0, 7501):
    records.append([str(data.values[i,j]) for j in range(0, 20)])
removed_records = []
for row in records:
    row = list(filter(lambda a: a != 'nan', row))
    row = list(filter(lambda a: a != 'mineral water', row))
    removed_records.append(row)

te = TransactionEncoder()
data = te.fit_transform(removed_records)
data = pd.DataFrame(data, columns = te.columns_)

from mlxtend.frequent_patterns import apriori,association_rules

frq_items = apriori(data, min_support = 0.004, use_colnames = True) 
rules = association_rules(frq_items, metric ="confidence", min_threshold = 0.2) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(len(rules))

global products
products = {"olive oil" : 'https://images-na.ssl-images-amazon.com/images/I/71JLJ0MQT8L._SY679_.jpg',
 "frozen vegetables" : 'https://images-na.ssl-images-amazon.com/images/I/81Dxf-0CzwL._SL1500_.jpg',
 "chocolate" : 'https://images-na.ssl-images-amazon.com/images/I/71cMqvY4T%2BL._SX569_.jpg',
 "spaghetti" : 'https://dq2y5jcmc9a4c.cloudfront.net/images/product/36/maicarspaghettipacket400g.jpg?t=1549059757',
 "eggs" : 'https://owldoor.com/wp-content/uploads/2020/01/rsz_12-dozen-eggs.jpg',
"burgers" : 'https://5.imimg.com/data5/ML/ST/ZZ/SELLER-11360013/mccain-420-gm-veggie-burgers-500x500.PNG',
 "green tea" : 'https://images-na.ssl-images-amazon.com/images/I/81Nop3iS8aL._SX569_.jpg',
 "cooking oil" : 'https://images-na.ssl-images-amazon.com/images/I/71VzUpmm5cL._SX425_.jpg',
 "milk" : 'https://static.turbosquid.com/Preview/2019/08/16__01_46_26/cover.jpg150AB9A0-3CE2-44BB-AC26-22AD50549874Large.jpg',
 "french fries" : 'https://5.imimg.com/data5/JE/JF/MY-63612283/mccain-french-fries-500x500.jpg',
 "ground beef" : 'https://i1.wp.com/www.eatthis.com/wp-content/uploads/2019/06/beyond-beef-ground-beef.jpg?w=640&ssl=1',
 "chicken" : 'https://5.imimg.com/data5/WW/AE/MY-1187642/chicken-breast-boneless-500x500.jpeg',
 "tomatoes" : 'https://www.therange.co.uk/_m5/4/1/1459913221_567.jpg',
  "pancakes" : 'https://images-na.ssl-images-amazon.com/images/I/71Jbc-PnpPL._SL1500_.jpg',}

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

app=Flask(__name__)

@app.route("/products", methods = ["GET"])
def get_products():
    res = []
    for key in products:
        res2 = {}
        value = products[key]
        res2["name"] = key
        res2["image"] = value
        res.append(res2)
    print(res)
    return jsonify(res)

@app.route("/results", methods = ["POST"])
def get_recommendation():
    # inp = {}
    inp = request.json['cartitems']
    print(inp)
    # hehe = '["olive oil","soup","frozen vegetables","eggs"]'
    # c = json.loads(inp)
    res = list(powerset(inp))
    s = []
    for combination in res:
        d = {''}
        for items in combination:
            d.add(items)
        d.remove('')
        s.append(d)
    total = []
    for subset in s:
        x = rules[ rules['antecedents'] == subset]
        total.append(x)
    recommendation = pd.concat(total)
    recommendation = recommendation.sort_values(['confidence', 'lift'], ascending =[False, False])
    print(len(recommendation))
    e = list(recommendation['consequents'])
    res1 = []
    for w in e:
        items = [x for x in w]
        res1.append(items[0])
    res1 = list(set(res1))
    res3 = []
    
    for key in res1:
        res2 = {}
        value = products[key]
        res2["name"] = key
        res2["image"] = value
        res3.append(res2)
    print(res2)
    return jsonify(res3)


if __name__ == '__main__':	
	app.debug=True
	app.run()













