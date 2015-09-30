import numpy as np 
import scipy
import pandas as pd
import cPickle as pickle  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import grid_search
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2

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


if __name__ == '__main__':
	print "reading the Chinese, English words and link types from file..."
	train = pd.read_csv('train.20k.pos.cln.features',header=0,delimiter=' ',quoting=3)

	'''
	print "extracting Chinese words..."
	chin_words = []
	for i in range(0,train["Chinese"].size):
		chin_words.append(train["Chinese"][i])

	print "extracting English words..."
	eng_words = []
	for i in range(0,train["English"].size):
		eng_words.append(train["English"][i])

	print "building Bag-of-words vector representation for Chinese words..."
	chinVectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
	chin_features = chinVectorizer.fit_transform(chin_words)
	chin_features = chin_features.toarray()
	outVec = open('chinVectorizer','wb')
	pickle.dump(chinVectorizer,outVec,pickle.HIGHEST_PROTOCOL)

	print "building Bag-of-words vector representation for English words..."
	engVectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
	eng_features = engVectorizer.fit_transform(chin_words)
	eng_features = eng_features.toarray()
	outVec2 = open('engVectorizer','wb')
	pickle.dump(engVectorizer,outVec2,pickle.HIGHEST_PROTOCOL)

	features = np.hstack([chin_features,eng_features])
	h = .02
	print "training Logistic Regression model..."
	model = linear_model.LogisticRegression(C=1e5)
	model = model.fit(features,train["Link"])
	outmodel = open('logisticRegressionModel','wb')
	pickle.dump(model,outmodel,pickle.HIGHEST_PROTOCOL)

	score = model.score(features,train["Link"])
	print "the score on the dev set is ", score

	print "split training data to dev and test set..."
	X_train, X_test, y_train, y_test = train_test_split(features, train["Link"], test_size=0.3, random_state=0)
	model2 = linear_model.LogisticRegression()
	model2 = model2.fit(X_train,y_train)
	score2 = model2.score(X_test,y_test)
	print "the score on test set is", score2

	print "split training data to do 10-fold cross validation..."
	scores = cross_val_score(linear_model.LogisticRegression(), features, train["Link"], scoring='accuracy', cv=10)
	print scores
	print "the average score for 10-fold cross validation is", scores.mean()
	'''

	'''
	pipeline = Pipeline([
    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
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
        },
    )),

    # Use a SVC classifier on the combined features
    ('LogisticRegression', linear_model.LogisticRegression()),
	])
	'''
	
	'''
	combined = FeatureUnion(
    transformer_list=[
        # Pipeline for pulling features from the post's subject line
        
        #('POS_Left2_cn', Pipeline([
        #    ('selector', ItemSelector(key='Left2_cn')),
        #    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        #])),
        ('POS_Left1_cn', Pipeline([
            ('selector', ItemSelector(key='Left1_cn')),
            ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        ])),
		

        ('POS_Chinese', Pipeline([
            ('selector', ItemSelector(key='POS_Chinese')),
            ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        ])),

        
        ('POS_Right1_cn', Pipeline([
            ('selector', ItemSelector(key='POS_Right1_cn')),
            ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        ])),
        #('POS_Right2_cn', Pipeline([
        #    ('selector', ItemSelector(key='Right2_cn')),
        #    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        #])),
		
        #('Left1_en', Pipeline([
        #    ('selector', ItemSelector(key='Left1_en')),
        #    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        #])),
		
        #('POS_Left1', Pipeline([
        #    ('selector', ItemSelector(key='POS_Left1')),
        #    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        #])),


        # Pipeline for standard bag-of-words model for body
        #('Prefix', Pipeline([
        #    ('selector', ItemSelector(key='Prefix')),
        #    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        #])),
        
        ('POS', Pipeline([
            ('selector', ItemSelector(key='POS')),
            ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        ])),

        #('POS_Right1', Pipeline([
        #    ('selector', ItemSelector(key='POS_Right1')),
        #    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        #])),

        #('Right1_en', Pipeline([
        #    ('selector', ItemSelector(key='Right1_en')),
        #    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        #])),
		

        #('POS_Right1', Pipeline([
        #    ('selector', ItemSelector(key='POS_Right1')),
        #    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        #])),


    ],

    # weight components in FeatureUnion
    transformer_weights={
    	#'POS_Left2_cn'  : 1.0,
    	'POS_Left1_cn' : 1.0,   	 
        'POS_Chinese': 1.0,
        'POS_Right1_cn': 1.0,
        #'POS_Right2_cn':1.0,
        #'Left1_en'  : 1.0,
        #'POS_Left1' : 1.0,
        #'Prefix': 1.0,
        'POS':1.0,
        #'POS_Right1':1.0,
        #'Right1_en':1.0,
        #'POS_Right1' : 1.0,
        })
        '''   
	combined = FeatureUnion(
		transformer_list=[
			('feature_type_1', FeatureUnion(
					transformer_list=[
						('Chinese', Pipeline([
            			('selector', ItemSelector(key='Chinese')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
                        ('Left1_en', Pipeline([
                        ('selector', ItemSelector(key='Left1_en')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Prefix', Pipeline([
                        ('selector', ItemSelector(key='Prefix')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
        				('POS', Pipeline([
            			('selector', ItemSelector(key='POS')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
					],
					transformer_weights={
						'Chinese':1.0,
						'Left1_en':1.0,
						'Prefix':1.0,
						'POS':1.0
					}
				)),
			('feature_type_2', FeatureUnion(
					transformer_list=[
						('Chinese', Pipeline([
            			('selector', ItemSelector(key='Chinese')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
        				('Right1_cn', Pipeline([
        				    ('selector', ItemSelector(key='Right1_cn')),
        				    ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
						('Right2_cn', Pipeline([
            			('selector', ItemSelector(key='Right2_cn')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
        				('English', Pipeline([
            			('selector', ItemSelector(key='English')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
					],
					transformer_weights={
						'Chinese':1.0,
						'Right1_cn':1.0,
						'Right2_cn':1.0,
						'English':1.0
					}
				)),
			('feature_type_3', FeatureUnion(
					transformer_list=[
						('Chinese', Pipeline([
            			('selector', ItemSelector(key='Chinese')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
						('Left1_en', Pipeline([
            			('selector', ItemSelector(key='Left1_en')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
        				('English', Pipeline([
            			('selector', ItemSelector(key='English')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
        				('Right1_en', Pipeline([
            			('selector', ItemSelector(key='Right1_en')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
					],
					transformer_weights={
						'Chinese':1.0,
						'Left1_en':1.0,
						'English':1.0,
						'Right1_en':1.0
					}
				)),
			('feature_type_4', FeatureUnion(
					transformer_list=[
						('Chinese', Pipeline([
            			('selector', ItemSelector(key='Chinese')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
        				('English', Pipeline([
            			('selector', ItemSelector(key='English')),
            			('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
        				])),
					],
					transformer_weights={
						'Chinese':1.0,
						#'Left1_en':1.0,
						'English':1.0,
						#'Right1_en':1.0
					}
				)),
            ('feature_type_5', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Left1', Pipeline([
                        ('selector', ItemSelector(key='POS_Left1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Prefix', Pipeline([
                        ('selector', ItemSelector(key='Prefix')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        'Chinese':1.0,
                        'POS_Left1':1.0,
                        'POS':1.0,
                        'Prefix':1.0
                    }
                )),
            ('feature_type_6', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Left1', Pipeline([
                        ('selector', ItemSelector(key='POS_Left1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('English', Pipeline([
                        ('selector', ItemSelector(key='English')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        'Chinese':1.0,
                        'POS_Left1':1.0,
                        'POS':1.0,
                        'English':1.0
                    }
                )),
            ('feature_type_7', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Prefix_Left1', Pipeline([
                        ('selector', ItemSelector(key='Prefix_Left1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Prefix', Pipeline([
                        ('selector', ItemSelector(key='Prefix')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Prefix_Right1', Pipeline([
                        ('selector', ItemSelector(key='Prefix_Right1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        'Chinese':1.0,
                        'Prefix_Left1':1.0,
                        'Prefix':1.0,
                        'Prefix_Right1':1.0
                    }
                )),
            ('feature_type_8', FeatureUnion(
                    transformer_list=[
                        ('Left1_cn', Pipeline([
                        ('selector', ItemSelector(key='Left1_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('English', Pipeline([
                        ('selector', ItemSelector(key='English')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        'Left1_cn':1.0,
                        'Chinese':1.0,
                        'English':1.0,
                        #'Prefix_Right1':1.0
                    }
                )),
            ('feature_type_9', FeatureUnion(
                    transformer_list=[
                        ('Left2_cn', Pipeline([
                        ('selector', ItemSelector(key='Left2_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Left1_cn', Pipeline([
                        ('selector', ItemSelector(key='Left1_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('English', Pipeline([
                        ('selector', ItemSelector(key='English')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        'Left2_cn':1.0,
                        'Left1_cn':1.0,
                        'Chinese':1.0,
                        'English':1.0,
                        #'Prefix_Right1':1.0
                    }
                )),
            ('feature_type_10', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Prefix', Pipeline([
                        ('selector', ItemSelector(key='Prefix')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'Chinese':1.0,
                        'Prefix':1.0,
                        #'Prefix_Right1':1.0
                    }
                )),
            ('feature_type_11', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Left1_en', Pipeline([
                        ('selector', ItemSelector(key='Left1_en')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('English', Pipeline([
                        ('selector', ItemSelector(key='English')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'Chinese':1.0,
                        'Left1_en':1.0,
                        'English':1.0,
                        #'Prefix_Right1':1.0
                    }
                )),
            ('feature_type_12', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('English', Pipeline([
                        ('selector', ItemSelector(key='English')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Right1_en', Pipeline([
                        ('selector', ItemSelector(key='Right1_en')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'Chinese':1.0,
                        #'Left1_en':1.0,
                        'English':1.0,
                        'Right1_en':1.0
                    }
                )),
            ('feature_type_13', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('English', Pipeline([
                        ('selector', ItemSelector(key='English')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'Chinese':1.0,
                        #'Left1_en':1.0,
                        'English':1.0,
                        'POS':1.0
                    }
                )),
            ('feature_type_14', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Left1', Pipeline([
                        ('selector', ItemSelector(key='POS_Left1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('English', Pipeline([
                        ('selector', ItemSelector(key='English')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Right1', Pipeline([
                        ('selector', ItemSelector(key='POS_Right1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'Chinese':1.0,
                        'POS_Left1':1.0,
                        'English':1.0,
                        'POS_Right1':1.0
                    }
                )),
            ('feature_type_15', FeatureUnion(
                    transformer_list=[
                        ('Chinese', Pipeline([
                        ('selector', ItemSelector(key='Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Left1', Pipeline([
                        ('selector', ItemSelector(key='POS_Left1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Right1', Pipeline([
                        ('selector', ItemSelector(key='POS_Right1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'Chinese':1.0,
                        'POS_Left1':1.0,
                        'POS':1.0,
                        'POS_Right1':1.0
                    }
                )),
            ('feature_type_16', FeatureUnion(
                    transformer_list=[
                        ('POS_Chinese', Pipeline([
                        ('selector', ItemSelector(key='POS_Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'POS_Chinese':1.0,
                        #'POS_Left1':1.0,
                        'POS':1.0,
                        #'POS_Right1':1.0
                    }
                )),
            ('feature_type_17', FeatureUnion(
                    transformer_list=[
                        ('POS_Chinese', Pipeline([
                        ('selector', ItemSelector(key='POS_Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Left1_en', Pipeline([
                        ('selector', ItemSelector(key='Left1_en')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('Prefix', Pipeline([
                        ('selector', ItemSelector(key='Prefix')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'POS_Chinese':1.0,
                        'Left1_en':1.0,
                        'Prefix':1.0,
                        'POS':1.0
                    }
                )),
            ('feature_type_18', FeatureUnion(
                    transformer_list=[
                        ('POS_Left1_cn', Pipeline([
                        ('selector', ItemSelector(key='POS_Left1_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Chinese', Pipeline([
                        ('selector', ItemSelector(key='POS_Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'POS_Chinese':1.0,
                        'POS_Left1_cn':1.0,
                        'POS':1.0,
                        #'POS_Right1':1.0
                    }
                )),
            ('feature_type_19', FeatureUnion(
                    transformer_list=[
                        ('POS_Left2_cn', Pipeline([
                        ('selector', ItemSelector(key='POS_Left2_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Left1_cn', Pipeline([
                        ('selector', ItemSelector(key='POS_Left1_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Chinese', Pipeline([
                        ('selector', ItemSelector(key='POS_Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS_Right1')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        'POS_Left1_cn':1.0,
                        'POS_Chinese':1.0,
                        'POS_Left2_cn':1.0,
                        'POS':1.0,
                        #'POS_Right1':1.0
                    }
                )),
            ('feature_type_20', FeatureUnion(
                    transformer_list=[
                        ('POS_Chinese', Pipeline([
                        ('selector', ItemSelector(key='POS_Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Right1_cn', Pipeline([
                        ('selector', ItemSelector(key='POS_Right1_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'POS_Chinese':1.0,
                        #'POS_Left1':1.0,
                        'POS':1.0,
                        'POS_Right1_cn':1.0
                    }
                )),
            ('feature_type_21', FeatureUnion(
                    transformer_list=[
                        ('POS_Chinese', Pipeline([
                        ('selector', ItemSelector(key='POS_Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Right1_cn', Pipeline([
                        ('selector', ItemSelector(key='POS_Right1_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Right2_cn', Pipeline([
                        ('selector', ItemSelector(key='POS_Right2_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        #'Left1_cn':1.0,
                        'POS_Chinese':1.0,
                        'POS_Right1_cn':1.0,
                        'POS':1.0,
                        'POS_Right2_cn':1.0
                    }
                )),
            ('feature_type_22', FeatureUnion(
                    transformer_list=[
                        ('POS_Left1_cn', Pipeline([
                        ('selector', ItemSelector(key='POS_Left1_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Chinese', Pipeline([
                        ('selector', ItemSelector(key='POS_Chinese')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS_Right1_cn', Pipeline([
                        ('selector', ItemSelector(key='POS_Right1_cn')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                        ('POS', Pipeline([
                        ('selector', ItemSelector(key='POS')),
                        ('bagOfWords', CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)),
                        ])),
                    ],
                    transformer_weights={
                        'POS_Left1_cn':1.0,
                        'POS_Chinese':1.0,
                        'POS_Right1_cn':1.0,
                        'POS':1.0,
                        #'POS_Right1':1.0
                    }
                )),
			
				
		],
		transformer_weights={
			'feature_type_1':1.0,
			'feature_type_2':1.0,
			'feature_type_3':1.0,
			'feature_type_4':1.0,
            'feature_type_5':1.0,
            'feature_type_6':1.0,
            'feature_type_7':1.0,
            'feature_type_8':1.0,
            'feature_type_9':1.0,
            'feature_type_10':1.0,
            'feature_type_11':1.0,
            'feature_type_12':1.0,
            'feature_type_13':1.0,
            'feature_type_14':1.0,
            'feature_type_15':1.0,
            'feature_type_16':1.0,
            'feature_type_17':1.0,
            'feature_type_18':1.0,
            'feature_type_19':1.0,
            'feature_type_20':1.0,
            'feature_type_21':1.0,
            'feature_type_22':1.0,
		}
		)
	
	print "extracting features..."
	X_train = combined.fit_transform(train)

	'''
	print "the original feature shape is", features.shape
	print "selecting the best 4 features..."
	best4 = SelectKBest(chi2, k=4).fit(features, train["Link"])
	print "the selected feature is", best4.get_support()
	print "the score for each feature is ", best4.scores_
	print "the p-value for each feature is ", best4.pvalues_

	X_train = best4.transform(features)
	print "new features shape is ", X_train.shape
	'''
	#print "dumping the word vector..."
	#combinedFeatureVec = open('combinedFeatureVec','wb')
	#pickle.dump(combined,combinedFeatureVec,pickle.HIGHEST_PROTOCOL)
	#model = linear_model.LogisticRegression(C=1,multi_class='multinomial',solver='lbfgs',penalty='l2')
	model = linear_model.LogisticRegression()
	#parameters = {'C':[1,10,100,1000]}
	#gridSearch = grid_search.GridSearchCV(model,parameters)
	#gridSearch.fit(features,train["Link"])
	#print (gridSearch.best_estimator_)
	print "fitting model..."
	model = model.fit(X_train,train["Link"])
	#print "dumping model..."
	#trainedLRModel = open('trainedLRModel','wb')
	#pickle.dump(model,trainedLRModel,pickle.HIGHEST_PROTOCOL)
	print "accuracy on dev set is", model.score(X_train,train["Link"])
	'''
	print "split training data to dev and test set..."
	X_train, X_test, y_train, y_test = train_test_split(features, train["Link"], test_size=0.3, random_state=0)
	model2 = linear_model.LogisticRegression()
	model2 = model2.fit(X_train,y_train)
	score2 = model2.score(X_test,y_test)
	print "the score on test set is", score2
	'''
	
	print "split training data to do 10-fold cross validation..."
	scores = cross_val_score(model, X_train, train["Link"], scoring='accuracy', cv=10)
	print scores
	print "the average score for 10-fold cross validation is", scores.mean()
	

'''
	features = combined.fit_transform(train)
	model = linear_model.LogisticRegression()
	model = model.fit(features,train["Link"])
	print model.score(features,train["Link"])
'''

'''
    features = combined.fit_transform(train)
    model = linear_model.LogisticRegression()
    model = model.fit(features,train["Link"])
    print model.score(features,train["Link"])
'''
	#pipeline.fit(train, train["Link"])
	#print pipeline.score(train, train["Link"])


