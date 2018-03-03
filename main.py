#!/usr/bin/env python

from datetime import datetime
from tabulate import tabulate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator
import argparse
import csv

def type_histogram(trs):

	ret_val = ''

	ret_val += '=========================' + '\n'
	ret_val += 'Type Histogram' + '\n'
	ret_val += '=========================' + '\n'

	# Initializing counters
	type_hist = {}
	for s in set([t['type'] for t in trs]):
		type_hist[s] = 0

	# Counting
	for t in trs:
		type_hist[t['type']] += t['pval']
		type_hist[t['type']] += t['nval']

	headers = ['Type','Value']
	body = [[t[0],t[1]] for t in sorted(type_hist.items(),key=operator.itemgetter(1),reverse=True)]
	ret_val += tabulate(body,headers=headers,tablefmt='simple',floatfmt=".2f")

	ret_val += '\n'
	ret_val += '=========================' + '\n'

	return ret_val

# Showing or saving to file
def show_or_store(outputs=[]):
	if not len(outputs):
		plt.show()
	else:
		for f in outputs:
			plt.savefig(f)

def gen_plot(trs,output_filename):

	# Plot params
	matplotlib.rcParams.update({'font.size': 8})
	matplotlib.rcParams.update({'axes.linewidth': 0.2})
	matplotlib.rcParams.update({'xtick.major.width': 0.2})
	matplotlib.rcParams.update({'ytick.major.width': 0.2})

	# Data
	date = [t['date'] for t in trs]
	bal = [t['bal'] for t in trs]
	pval = [t['pval'] for t in trs]
	nval = [t['nval'] for t in trs]

	# Outlier values
	lpval = filter_thres(trs,'pval')
	lnval = filter_thres(trs,'nval')

	# Removing zeros
	for i,v in enumerate(nval):
		if v == 0:
			nval[i] = float('nan')
	for i,v in enumerate(pval):
		if v == 0:
			pval[i] = float('nan')

	# Plotting and labeling
	plt.figure(figsize=(10,7))
	plt.plot(date,bal)
	plt.scatter(date,pval, marker='+', color='g', s=80)
	plt.scatter(date,nval, marker='x', color='r')
	plt.xlabel('Time')
	plt.ylabel('Amount ($)')
	plt.axis((min(date),max(date),min(bal+nval),max(bal+pval)))
	plt.grid()
	plt.legend(['Balance','Income','Expenditure'],loc='best')

	# Labelling outlier values
	for l in lpval:
		plt.annotate(l['desc'].split(' ', 1)[0],(l['date'],l['pval']))
	for l in lnval:
		plt.annotate(l['desc'].split(' ', 1)[0],(l['date'],l['nval']))

	show_or_store(output_filename)


# Mean + Std
def get_thres(arr):
	mean = np.mean(arr)
	std = np.std(arr)
	if mean < 0:
		std *= -1
	return mean+2*std

def total_small(trs,entry):

	# Computing the threshold
	lst = [t[entry] for t in trs]
	thres = get_thres(lst)

	# Filtering dictionary
	return sum([t[entry] for t in trs if abs(t[entry]) < abs(thres)])

def filter_thres(trs,entry):

	# Computing the threshold
	lst = [t[entry] for t in trs]
	thres = get_thres(lst)

	# Filtering dictionary
	return [t for t in trs if abs(t[entry]) > abs(thres)]


# Main
def main():

	# Parsing user input
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'-i','--input',
			nargs='?',
			required=True,
			help='Input file name.'
		)
	parser.add_argument(
			'-o','--output',
			nargs='*',
			required=True,
			help='Output file name.'
		)
	args = parser.parse_args()

	trs = []
	with open(args.input, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		reader.next()
		for i,row in enumerate(reversed(list(reader))):
			try:
				trs += [{
					'index': i,
					'date': datetime.strptime(row[1], '%m/%d/%Y'),
					'desc': row[2],
					'pval': [max(t,0) for t in [float(row[-5])]][0],
					'nval': [min(t,0) for t in [float(row[-5])]][0],
					'type': row[-4],
					'bal': float(row[-3]),
				}]
			except:
				continue

	print type_histogram(trs)
	print 'Total small positive: %9.3f' % total_small(trs,'pval')
	print 'Total small negative: %9.3f' % total_small(trs,'nval')
	gen_plot(trs,args.output)


if __name__ == "__main__":
    main()
