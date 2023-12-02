from django.shortcuts import render
#ml libs
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib.pyplot as plt

def index(request):
    return render(request,'index.html')

def show(request):
    df = pd.read_csv('C:/Users/User/OneDrive/Desktop/bidding/1_prepared_data.csv')
    auction_df = df.groupby(['auction_id', 'open_price', 'closing_price', 'item_id', 'auction_type'], observed=True).agg(num_bids=('bidder_id', pd.Series.nunique)).reset_index()
    auction_df = auction_df.drop(columns=['auction_id'])
    auction_df.head()
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score

    X = auction_df.drop(columns=['closing_price', 'num_bids'], axis=1)
    y = auction_df['num_bids']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lr = LinearRegression()
    lr.fit(X,y)

    value1 = int(request.GET['open_price'])
    value2 = int(request.GET['item_id'])
    value3 = int(request.GET['auction_type'])
    predct= lr.predict([[value1,value2,value3]])

    data={'v1':value1,
          'v2':value2,
          'v3':value3,
          'prdct' :predct[0]
          }
    
    
    return render(request,'show.html',data)
