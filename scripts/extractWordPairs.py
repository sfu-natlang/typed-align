from __future__ import division
import sys, math
from itertools import *
import cPickle as pickle
import codecs

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
		old = sys.stdout
		sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
		chin = open(sys.argv[1],'rb')
		eng = open(sys.argv[2],'rb')
		gold = open(sys.argv[3],'rb')
		#output = open(sys.argv[2],'w')
		for line in chin:
			utf8line = unicode(line,'utf-8')
			chinChars = utf8line.split()
			engWords = unicode(eng.readline(),'utf-8').split()
			alignments = gold.readline().split()
			for alm in alignments:
				if not alm[0].isdigit():#alm[0]=='-':
					continue
				left = alm.split('-')[0]
				#right = alm.split('-')[1].split
				if isContiguous(left):
					chinWord = ""
					charIndices = strip(left)
					for charIndex in charIndices:
						chinWord += chinChars[int(charIndex)-1]
					Left1_cn = chinChars[int(charIndices[0])-2] if int(charIndices[0]) > 1 else 'null#null' 
					Left2_cn = chinChars[int(charIndices[0])-3] if int(charIndices[0]) > 2 else 'null#null'
					Right1_cn = chinChars[int(charIndices[len(charIndices)-1])] if int(charIndices[len(charIndices)-1]) < len(chinChars) else 'null#null'
					Right2_cn = chinChars[int(charIndices[len(charIndices)-1])+1] if int(charIndices[len(charIndices)-1]) < len(chinChars)-1 else 'null#null'
					right = alm.split('-')[1]
					linkLabel = right[-4:-1]
					engIndices = strip(right[:-5])
					if engIndices[0].strip():
						for wordIndex in engIndices:
							words = engWords[int(wordIndex)-1].split('_')
							Left1_en = engWords[int(wordIndex)-2] if int(wordIndex) > 1 else 'null_null'
							Right1_en = engWords[int(wordIndex)] if int(wordIndex) < len(engWords) else 'null_null'
							print Left2_cn.split('#')[0],Left2_cn.split('#')[1],Left1_cn.split('#')[0],Left1_cn.split('#')[1],chinWord.split('#')[0],chinWord.split('#')[1],Right1_cn.split('#')[0],Right1_cn.split('#')[1],Right2_cn.split('#')[0],Right2_cn.split('#')[1],Left1_en.split('_')[0],Left1_en.split('_')[0][:5],Left1_en.split('_')[1],words[0],words[0][:5],words[1],Right1_en.split('_')[0],Right1_en.split('_')[0][:5],Right1_en.split('_')[1],linkLabel
			#tmp = list()
			#for word in words:
			#tmp.append(word)
			#print " ".join(tmp)
		sys.stdout = old	
