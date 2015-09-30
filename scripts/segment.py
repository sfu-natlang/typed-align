# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import sys, math
from itertools import *
import cPickle as pickle
import codecs
import re


def isContiguous(chin):
	left = strip(chin)
	if len(left)==1:
		return True
	#left = chin.split(',')
	for i in range(1,len(left)):
		if int(left[i])-int(left[i-1])>1:
			return False
	return True

def strip(eng):
	indices = str()
	for char in eng:
		if char.isdigit() or char==',':
			indices += char
	return indices.split(',')

if __name__=='__main__':
	if len(sys.argv) > 1:
		reload(sys)
		sys.setdefaultencoding('utf-8')
		old = sys.stdout
		sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
		chin = open(sys.argv[1],'rb')
		#eng = open(sys.argv[2],'rb')
		gold = open(sys.argv[2],'rb')
		chinOut = open(sys.argv[3],'w')
		goldOut = open(sys.argv[4],'w')
		for line in chin:
			utf8line = unicode(line,'utf-8')
			chinChars = utf8line.split()
			wa = gold.readline().split()
			if wa[0].isalpha():# == "rejected":
				print("this line is abandoned ",file=chinOut)
				print("rejected",file=goldOut)
				continue
			alignments = sorted(wa, key=lambda x:[int(s) for s in re.findall(r'\d+',x.split('-')[0])][0] if x.split('-')[0]!='' else 0)
			wordIndex = 0
			lastIndex = 0
			segments = []
			wordWA = [] 
			for alm in alignments:
				if not alm[0].isdigit():#alm[0]=='-':
					continue
				left = alm.split('-')[0]
				right = alm.split('-')[1]
				if isContiguous(left):
					chinWord = ""
					charIndices = strip(left)
					if(int(charIndices[0])>lastIndex+1):
						word = ""
						for index in range(lastIndex+1,int(charIndices[0])):
							word += chinChars[int(index)-1]
						segments.append(word)
						wordIndex += 1
						lastIndex = int(charIndices[0])-1
					for charIndex in charIndices:
						chinWord += chinChars[int(charIndex)-1]
					segments.append(chinWord)
					wordIndex += 1
					wordWA.append(str(wordIndex)+'-'+right)
					lastIndex = int(charIndices[len(charIndices)-1])
				else:
					chinWord = chinChars[int(strip(left)[0])-1]
					segments.append(chinWord)
					wordIndex += 1
					lastIndex = int(int(strip(left)[0]))
			print(' '.join(segments),file=chinOut)
			print(' '.join(wordWA),file=goldOut)
			#tmp = list()
			#for word in words:
			#tmp.append(word)
			#print " ".join(tmp)
		sys.stdout = old	
