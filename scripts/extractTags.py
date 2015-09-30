from __future__ import division
import sys, math
from itertools import *
import cPickle as pickle

if __name__ == "__main__":
	text = open(sys.argv[1], 'rb')
	for line in text:
		tags = [word_tag.split('*=*')[1] for word_tag in line.split()]
		print ' '.join(tags)
