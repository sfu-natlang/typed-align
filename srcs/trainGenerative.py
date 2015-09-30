from __future__ import division
import sys, math
from itertools import *
import cPickle as pickle
#import numpy as np 
#import scipy
#import pandas as pd
#import cPickle as pickle  
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn import linear_model
#from sklearn.cross_validation import train_test_split
#from sklearn.cross_validation import cross_val_score
#from sklearn.pipeline import FeatureUnion
#from sklearn.pipeline import Pipeline
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn import grid_search

#"Helper argmax function"
def argmax(ls):
	if not ls: return None, 0.0
	return max(ls, key = lambda x: x[1])

#"Helper count increment function, could also use defaultdict."
def inc(d, k, delta):
	d.setdefault(k, 0.0)
	d[k] += delta

class Counts:
#"Collect the counts c in the pseudo code"
	def __init__(self):
		# c(e)
		self.word = {}
		# c(e, f)
		self.words = {}
		# for c(e,f,h)
		self.link = {}
		# For model 2.
		# c(i, l, m)
		#self.align = {}
		# c(j, i, l, m)
		#self.aligns = {}


class AlignmentModel:
#"Base alignment model. Stores counts, gives translation parameters "
	def reestimate(self, counts):
	#"Compute the MLE estimate based on counts."
		self.counts = counts
		self.first_run = False
	
	def t(self, f, e):
	#"The t(f|e) parameter."
		c = self.counts.words[e, f] if (e, f) in self.counts.words else 0
		n = self.counts.word[e] if e in self.counts.word else 0
		v = 163303
		alpha = 0.005
		return (c + alpha) / (n + alpha * v)
		#if (e, f) in self.counts.words:
		#	return self.counts.words[e, f] / self.counts.word[e]
		#else:
		#	return 1e-12

	def s(self,e,f,h):
		if (e,f,h) in self.counts.link:
			return self.counts.link[e,f,h] / self.counts.words[e,f]
		else:
			tagDist = {'SEM':0.401, 'FUN':0.246, 'PDE':0.004, 'CDE':0.004, 'MDE':0.012, 'GIS':0.205, 'GIF':0.031, 'COI':0.008, 'TIN':0.003, 'NTR':0.086, 'MTA':0.002}
			return tagDist[h]
	
	def align(self, e, f):
	#"Computes the best alignment from a model and two sentences."
		#l = len(e)
		#m = len(f)
		alignment = []
		links = ['SEM', 'FUN', 'GIS', 'GIF', 'COI', 'PDE', 'CDE', 'MDE', 'TIN', 'NTR', 'MTA']
		for i, f_i in enumerate(f):
			j, p, l = argmax([(j, self.h(e_j, f_i, l), l) for j, e_j in enumerate(e) for l in links])
			alignment.append((j,l))
		return alignment

	def p(self, e, f):
#	"The probability of alignment. Will combine parameters."
		pass

class Model1(AlignmentModel):
	def __init__(self, counts):
	#"Initialize the alignment model"
		self.first_run = True
		self.counts = counts

	def p(self, e, f): return self.t(f, e)

	def h(self, e, f, h): return self.t(f,e) * self.s(e,f,h)

def findAlignment(wordModel, tagModel, e, f):
	alignment = []
	links = ['SEM', 'FUN', 'GIS', 'GIF', 'COI', 'PDE', 'CDE', 'MDE', 'TIN', 'NTR', 'MTA']
	for i, f_i in enumerate(f):
		f_word = f_i.split('_')[0]
		f_tag = f_i.split('_')[1]
		candidates = []
		#j, p, l = argmax([(j, wordModel.t(f_i.split('_')[0],e_j.split('#')[0])*(alpha*wordModel.s(e_j.split('#')[0], f_i.split('_')[0], l)+(1-alpha)*tagModel.s(e_j.split('#')[1], f_i.split('_')[1], l)), l) for alpha in [wordModel.counts.words[e_j.split('#')[0], f_i.split('_')[0]] if (e_j.split('#')[0], f_i.split('_')[0]) in wordModel.counts.words else 0 / (wordModel.counts.words[e_j.split('#')[0], f_i.split('_')[0]] if (e_j.split('#')[0], f_i.split('_')[0]) in wordModel.counts.words else 0 + tagModel.counts.words[e_j.split('#')[1], f_i.split('_')[1]] if (e_j.split('#')[1], f_i.split('_')[1]) in tagModel.counts.words else 0)  ] for j, e_j in enumerate(e) for l in links])
		for j, e_j in enumerate(e):
			for l in links:
				e_word = e_j.split('#')[0]
				e_tag = e_j.split('#')[1]
				if (e_word, f_word) in wordModel.counts.words and (e_tag, f_tag) in tagModel.counts.words:
					alpha = wordModel.counts.words[e_word, f_word] / (wordModel.counts.words[e_word, f_word] + tagModel.counts.words[e_tag, f_tag])
					p = wordModel.t(f_word,e_word)*(alpha*wordModel.s(e_word, f_word, l) + (1-alpha)*tagModel.s(e_tag, f_tag, l))
				elif (e_tag, f_tag) in tagModel.counts.words:
					p = wordModel.t(f_word,e_word)*tagModel.s(e_tag, f_tag, l)
				elif (e_word, f_word) in wordModel.counts.words:
					p = wordModel.t(f_word,e_word)*wordModel.s(e_word, f_word, l)
				else:
					p = 1e-13
				candidates.append((j,p,l))
		
		j, p, l = argmax(candidates)
		alignment.append((j,l))
	return alignment
'''
def concatenate(f, e, flag):
	if(flag):
		return {'Chinese':[e],'English':[f]}
	else:
		return {'Chinese':[f],'English':[e]}

def jointProb(model, e_j, f_i, vectorModel, LogitRegModel, flag):
	return model.p(e_j, f_i) * max(LogitRegModel.predict_proba(vectorModel.transform(concatenate(f_i, e_j, flag)))[0])

def alignWithLogitReg(model, e, f, vectorModel, LogitRegModel, flag):
#"Computes the best alignment from a model and two sentences."
		l = len(e)
		m = len(f)
		alignment = []
		link = []
		for i, f_i in enumerate(f):
			j, p, l = argmax([(j, jointProb(model,e_j,f_i,vectorModel,LogitRegModel,flag), LogitRegModel.predict(vectorModel.transform(concatenate(f_i, e_j, flag)))[0]) for j, e_j in enumerate(e)])
			alignment.append(j)
			link.append(l)
		result = {'alignment':alignment, 'link':link}
		return result
'''

def EM(counter, model, iterations = 5):
#"EM algorithm. Relies on counter to do the heavy lifting."
	for k in range(iterations):
		print "begin EM algorithm iteration #", k
		sys.stdout.flush()
		counts = counter.expected_counts(model)
		model.reestimate(counts)
	return model

def isContiguous(chin):
	#left = strip(chin)
	#if len(left)==1:
	#	return True
	#left = chin.split(',')
	#for i in range(1,len(left)):
	#	if int(left[i])-int(left[i-1])>1:
	#		return False
	return True

def strip(eng):
	indices = str()
	for char in eng:
		if char.isdigit() or char==',':
			indices += char
	return indices.split(',')

class Counter:
#"Class for computing the expected counts of both models."
	def __init__(self, english_corpus, french_corpus, gold_alignment):
		self.tritext = zip(english_corpus, french_corpus, gold_alignment)

	def init_model(self):
	# For initializing the counts n(e). Each word type gets 1 count.
		print "begin to initialize model"
		initial_counts = Counts()
		index = 0
		for e, f, wa in self.tritext:
			if index % 1000 == 0:
                		print "processing sentence ", index
                		sys.stdout.flush()
			for e_j in e:
				for f_i in f:
					key = (e_j, f_i)
					if key not in initial_counts.words:
						initial_counts.words[key] = 1.0
						inc(initial_counts.word, e_j, 1.0)
			for alm in wa:
				if not alm[0].isdigit():
					if alm[0] != '-':
						continue
					chinWord = "*"
					right = alm.split('-')[0]
					linkLabel = right[-4:-1]
					engIndices = strip(right[:-5])
					if engIndices[0].strip():
						for wordIndex in engIndices:
							engWord = f[int(wordIndex)-1]
							if (chinWord, engWord) in initial_counts.words:
								inc(initial_counts.link, (chinWord, engWord, linkLabel), 1.0)
				else:
					left = alm.split('-')[0]
					if isContiguous(left):
						#chinWord = ""
						#charIndices = strip(left)
						#for charIndex in charIndices:
						#	chinWord += c[int(charIndex)]
						chinWord = e[int(left)]
						right = alm.split('-')[1]
						linkLabel = right[-4:-1]
						engIndices = strip(right[:-5])
						if engIndices[0].strip():
							for wordIndex in engIndices:
								engWord = f[int(wordIndex)-1]
								if (chinWord, engWord) in initial_counts.words:
									inc(initial_counts.link, (chinWord, engWord, linkLabel), 1.0)
			index += 1
		print "finish initializing model"
		return initial_counts

	def expected_counts(self, model):
	#"The main algorithm from the notes. Used for both model1 and model2."
		print "begin to calculate expected_counts"
		counts = Counts()
		for s, (e, f, wa) in enumerate(self.tritext):
			if s % 1000 == 0:
                		print "calculate sentence pair #", s
                		sys.stdout.flush()
			#l = len(e)
			#m = len(f)
			for f_i in f :#for i, f_i in enumerate(f):
				total = sum((model.p(e_j, f_i) for e_j in e)) #(j, e_j) in enumerate(e)))
				for e_j in e: #for j, e_j in enumerate(e):
					delta = model.p(e_j, f_i) / total
					inc(counts.word, e_j, delta)
					inc(counts.words, (e_j, f_i), delta)
					# Only used in model 2.
					#inc(counts.aligns, (j, i, l, m), delta)
					#inc(counts.align, (i, l, m), delta)
				for alm in wa:
					if not alm[0].isdigit():
						if alm[0] != '-':
							continue
						chinWord = "*"
						right = alm.split('-')[0]
						linkLabel = right[-4:-1]
						engIndices = strip(right[:-5])
						if engIndices[0].strip():
							for wordIndex in engIndices:
								engWord = f[int(wordIndex)-1]
								if (chinWord, engWord) in model.counts.words:
									delta = model.h(chinWord,engWord,linkLabel) / total
									inc(counts.link, (chinWord,engWord,linkLabel), delta)
					else:
						left = alm.split('-')[0]
						if isContiguous(left):
							#chinWord = ""
							#charIndices = strip(left)
							#for charIndex in charIndices:
							#	chinWord += c[int(charIndex)]
							chinWord = e[int(left)]
							right = alm.split('-')[1]
							linkLabel = right[-4:-1]
							engIndices = strip(right[:-5])
							if engIndices[0].strip():
								for wordIndex in engIndices:
									engWord = f[int(wordIndex)-1]
									if (chinWord, engWord) in model.counts.words:
										delta = model.h(chinWord,engWord,linkLabel) / total
										inc(counts.link, (chinWord,engWord,linkLabel), delta)
		return counts
'''
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

def getFeatureUnion():
	return FeatureUnion(
    	transformer_list=[
        	# Pipeline for pulling features from the post's subject line
        	('Chinese', Pipeline([
            	('selector', ItemSelector(key='Chinese')),
            	('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        	])),

        	# Pipeline for standard bag-of-words model for body
        	('English', Pipeline([
            	('selector', ItemSelector(key='English')),
            	('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        	])),
    	],

    	# weight components in FeatureUnion
    	transformer_weights={
        	'Chinese': 1.0,
        	'English': 1.0,
        	})
'''

def read_corpus(english_corpus, french_corpus, gold_alignment):
#"Reads a corpus and adds in the NULL word."
	print "reading corpus..."
	english = [["null"] + e_sent.split() for e_sent in open(english_corpus)]
	french = [f_sent.split() for f_sent in open(french_corpus)]
	#chinese_char = [["*"] + c_sent.split() for c_sent in open(chinese_char)]
	gold_wa = [wa_sent.split() for wa_sent in open(gold_alignment)]
	return english, french, gold_wa

def read_corpus2(english_corpus, french_corpus):
#"Reads a corpus and adds in the NULL word."
	print "reading corpus..."
	english = [["*#null"] + e_sent.split() for e_sent in open(english_corpus)]
	french = [f_sent.split() for f_sent in open(french_corpus)]
	return english, french

def main(mode):
	if mode == "train":
		english, french, gold_alignment = read_corpus('train.20k.tags.cn', 'train.20k.tags.en', 'train.20k.seg.cln.wa')
		counter = Counter(english, french, gold_alignment)
		model1 = Model1(counter.init_model())
		model1 = EM(counter, model1)
		out_model = open('model.20k.tags.wordLink.cn-en', 'wb')
		pickle.dump(model1, out_model, pickle.HIGHEST_PROTOCOL)
	elif mode == "align":
		print "loading alignment model..."
		wordModel = pickle.load(open('model.hk+20k.wordLink.cn-en', 'rb'))
		tagModel = pickle.load(open('model.hk+20k.tags.wordLink.cn-en', 'rb'))
		english, french = read_corpus2('test.seg.cln.pos.cn', 'test.seg.cln.pos.en')
		print "finding best alignment..."
		for s, (e, f) in enumerate(zip(english, french), 1):
			alignment = findAlignment(wordModel,tagModel,e,f) #model.align(e, f)
			for i, a_i in enumerate(alignment, 1):
				print s, a_i[0], i, a_i[1]
	elif mode == "alignWithLogitReg":
		print "loading alignment model..."
		model = pickle.load(open('model.word.en-cn', 'rb'))
		english, french = read_corpus('dev.tok.cn', 'dev.lc.en')
		#print "reading the link data..."
		#train = pd.read_csv('train.cln.features',header=0,delimiter=' ',quoting=3)
		#combined = getFeatureUnion()
		print "transforming features..."
		#features = combined.fit_transform(train)
		combined = pickle.load(open('combinedFeatureVec','rb'))
		print "loading LogitRegModel..."
		logitRegModel = pickle.load(open('trainedLRModel','rb'))
		print "finding best alignment..."
		for s, (e, f) in enumerate(zip(english, french), 1):
			result = alignWithLogitReg(model, e, f, combined, logitRegModel, True)
			for i, (a_i, l) in enumerate(zip(result['alignment'], result['link']), 1):
				print s, a_i, i, l
	elif mode == "intersect":
		model = pickle.load(open('model.char.cn-en', 'rb'))
		model_reverse = pickle.load(open('model.char.en-cn', 'rb'))
		english, french = read_corpus('dev.cn', 'dev.en')
		train = pd.read_csv('train.cln.features',header=0,delimiter=' ',quoting=3)
		combined = getFeatureUnion()
		features = combined.fit_transform(train)
		for s, (e, f) in enumerate(zip(english, french), 1):
			#print "for sentence", s
			alignment = model.align( e, f)
			alignment2 = model_reverse.align(["*"] + f, e[1:])
			align1 = set(enumerate(alignment, 1))
			align2 = set([ (j,i) for i, j in enumerate(alignment2, 1)])
			intersect = align1 & align2
			for i, a_i in intersect:
				print s, a_i, i
	elif mode == "gdf":
		model = pickle.load(open('model.char.cn-en', 'rb'))
		model_reverse = pickle.load(open('model.char.en-cn', 'rb'))
		english, french = read_corpus('dev.cn', 'dev.en')
		for s, (e, f) in enumerate(zip(english, french), 1):
			#print "for sentence", s
			alignment = model.align( e, f)
			alignment2 = model_reverse.align(["*"] + f, e[1:])
			align1 = set(enumerate(alignment, 1))
			align2 = set([ (j,i) for i, j in enumerate(alignment2, 1)])
			intersect = align1 & align2
			union = align1 | align2
			neighbors = [(-1,0), (0,-1), (1,0), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
			m = len(e)
			n = len(f)
			#grow-diag
			#print "initial intersect is :", intersect
			while True:
				set_len = len(intersect)
				for e_word in xrange(1, m+1):
					for f_word in xrange(1, n+1):
						if (e_word, f_word) in intersect:
							for (e_diff, f_diff) in neighbors:
								e_new = e_word + e_diff
								f_new = f_word + f_diff
								if not intersect:
									if (e_new, f_new) in union:
										intersect.add((e_new, f_new))
								else:
									if ((e_new not in zip(*intersect)[0] or f_new not in zip(*intersect)[1]) and ((e_new, f_new) in union)):
										#print intersect
										#print (e_new, f_new)
										intersect.add((e_new, f_new))

				if  set_len == len(intersect):
					break
			#final
			
			for e_word in xrange(1, m+1):
				for f_word in xrange(1, n+1):
					if not intersect:
						if (e_word, f_word) in union:
							intersect.add((e_word, f_word))
					else:
						if ((e_word not in zip(*intersect)[0] or f_word not in zip(*intersect)[1]) and ((e_word, f_word) in union)):
							intersect.add((e_word, f_word))
			
			#print intersect
			for i, a_i in intersect:
				print s, a_i, i
	elif mode == "convert":
		align = open(sys.argv[2], 'rb')
		giza = open(sys.argv[3], 'w')
		linum = 1;
		for line in align:
			num = int(line.split()[0])
			if (num != linum):
				if(num!=linum+1):
					giza.write("\n")
					linum += 1
					continue
				else:
					linum = linum + 1
					giza.write("\n")
			a = str(int (line.split()[1])) + "-" +  str(int (line.split()[2]))
			giza.write(a)
			giza.write(" ")
	elif mode == "convertGold":
		align = open(sys.argv[2], 'rb')
		giza = open(sys.argv[3], 'w')
		for line in align:
			for alms in line.split():
				stripped = ""
				for letter in alms:
					if (letter.isdigit() or letter==',' or letter=='-'):
						stripped += letter
				if(len(stripped)==0):
					continue
				if(stripped[0]=='-'):
					stripped = '0'+stripped
				elif(stripped[len(stripped)-1]=='-'):
					stripped += '0'
				giza.write(stripped)
				giza.write(" ")
			giza.write("\n")
	elif mode == "convertWord":
		chin = open(sys.argv[2], 'rb')
		align = open(sys.argv[3], 'rb')
		eng = open(sys.argv[4], 'rb')
		giza = open(sys.argv[5], 'w')
		for line in chin:
			words = line.split()
			count = {}
			index = 1
			wordCount = 1
			for word in words:
				if (word.isalnum() or len(word)%3!=0):
					count[wordCount] = str(int(index))
					index += 1
					wordCount += 1
					continue
				chars = list()
				for i in range(index,index+len(word)/3):
					chars.append(str(int(i)))
				count[wordCount] = ','.join(chars)
				wordCount += 1
				index += len(word)/3
			alms = align.readline()
			count_e = len(eng.readline().split())
			if(len(alms.split())==0):
				for i in range(1,count_e+1):
					giza.write('0-'+str(i))
					giza.write(' ')
				giza.write("\n")
				continue
			for alm in alms.split():
				left = alm.split('-')[0]
				if left!='0':
					left = count[int(left)]
				giza.write(left+'-'+alm.split('-')[1])
				giza.write(' ')
			giza.write("\n")
	elif mode == "check":
		align = open(sys.argv[2], 'rb')
		alms_count = 0
		chin_count = 0
		eng_count = 0
		chin_eng_count = 0
		normal_count = 0
		for line in align:
			alignments = line.split()
			for alms in alignments:
				alms_count += 1
				left = alms.split('-')[0].split(',')
				right = alms.split('-')[1].split(',')

				diff = 0
				for i in range(0,len(left)):
					if(i>0):
						diff += int(left[i])-int(left[i-1])
				sub = 0
				for i in range(0,len(right)):
					if(i>0):
						sub += int(right[i])-int(right[i-1])
				if(diff > len(left)-1):
					if(sub>len(right)-1):
						chin_eng_count += 1
					else:
						chin_count += 1
				else:
					if(sub>len(right)-1):
						eng_count += 1
					else:
						normal_count += 1

		print alms_count
		print normal_count
		print chin_count
		print eng_count
		print chin_eng_count
		





if __name__ == "__main__":
	main(sys.argv[1])




