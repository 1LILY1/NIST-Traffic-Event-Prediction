import pandas as pd
from datetime import timedelta
import numpy as np
import time
import datetime
import math
from multiprocessing import Pool
from functools import partial

start_time = time.time()
predDF = pd.read_csv("./prediction_trials.tsv", header=0, sep='\t')
events = pd.read_csv("./events_train.tsv", header=0, sep='\t')
events['Year'] = pd.DatetimeIndex(events['closed_tstamp']).year
a=events[events["event_type"]=="accidentsAndIncidents"]
r=events[events["event_type"]=="roadwork"]
p=events[events["event_type"]=="precipitation"]
d=events[events["event_type"]=="deviceStatus"]
o=events[events["event_type"]=="obstruction"]
t=events[events["event_type"]=="trafficConditions"]
l=[2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
n=dict.fromkeys(l, 0)
tlist = []
for i in range(len(predDF)):
	tlist.append(predDF.loc[i])
	
df=pd.DataFrame()



def f(pts, row):
	x1, x2 = row['nw_lon'], row['se_lon']
	y1, y2 = row['se_lat'], row['nw_lat']
	ll = np.array([x1, y1])  # lower-left
	ur = np.array([x2, y2])  # upper-right
	inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
	inbox = pts[inidx]
	return len(inbox)

if __name__ == '__main__':
    
	for i in l:
		p = Pool(3)
		b=a[a["Year"]==i]
		pts = b[['longitude', 'latitude']]
		pts = pts.as_matrix()
		func = partial(f, pts)
		outList = p.map(func, tlist[:])
		df[i]= outList
		df.to_csv("alek2.csv")
	print time.time() - start_time
