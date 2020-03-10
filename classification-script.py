from __future__ import print_function
from __future__ import division
 
import pandas as pd
from collections import defaultdict
import numpy as np
import os
import sklearn
import sklearn.multiclass
import sklearn.svm
import sklearn.model_selection
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.externals.joblib
import sklearn.metrics
import sklearn.feature_extraction.text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from gensim import corpora
from keras.models import Sequential, Model
from keras.initializers import Constant
from sklearn.utils.class_weight import compute_class_weight
from os.path import join, exists, split
from gensim.models import word2vec
from gensim.models import Phrases
import gensim
import scipy.stats as st
from keras import regularizers
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold
reload(sys)
sys.setdefaultencoding('utf8')

import ConfigParser
import scipy

# NN parameters:
embedding_dims =500
patience = 10
initializer_func = 'random_uniform'


filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 35
epochs = 1000

# global maxlen
# maxlen = 487
np.random.seed(8)

def extract_featuresLDA(X, val_X, test_X, max_ngram_size=2):
 
    def get_lda(X, dictionary):
        X_res = []
        for x in X:
            new_vec = dictionary.doc2bow(x)
            doc_lda = lda.get_document_topics(new_vec, minimum_probability=0.0)
            dt = np.dtype('int,float')
            xarr = np.array(doc_lda, dtype=dt)
            X_res.append(xarr['f1'])
        return np.array(X_res)

 
    X_texts = X
    val_X_texts = val_X
    test_X_texts = test_X
    
    texts = X_texts + val_X_texts + test_X_texts
    
       
    supp_txt = [line.rstrip('\n').split(' ') for line in open('train.txt')]
    # supp_txt = make_text_list(supp_txt)

    texts = supp_txt

    f = codecs.open('nltk_stopwords.txt', 'r', encoding="utf-8")
    stoplist = f.read().split()
   
    dictionary = corpora.Dictionary(texts)
    # remove stop words and words that appear only once
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
    dictionary.filter_tokens(stop_ids)
    dictionary.compactify()
    dictionary.save('lda.dict')
 
    corpus = [dictionary.doc2bow(text) for text in texts]
    id2token = {v: k for k, v in dictionary.token2id.items()}
    lda = gensim.models.ldamodel.LdaModel(corpus, id2word=id2token, num_topics=150, passes=1)
    lda.save('model.lda')
  
    vect = ''
    X_features = get_lda(X_texts, dictionary)
    val_X_features = get_lda(val_X_texts, dictionary)
    test_X_features = get_lda(test_X_texts, dictionary)

    return vect, X_features, val_X_features, test_X_features

def make_predictionsLDA(X, Y, val_X, val_Y, test_X, test_Y, s, test_ids):
    
    classifier = RandomForestClassifier(class_weight="balanced_subsample", max_depth=2)
 
    total_data = X.shape[0]

    classifier.fit(X, Y)
    test_y_hat = classifier.predict(test_X)

    prec, recall, fm, support = precision_recall_fscore_support(test_Y, test_y_hat)
    print('F-measure')
    print(fm)
    print('Precision')
    print(prec)
    print('Recall')
    print(recall)
    print('Stat')
    print(support)

    accuracy_score = sklearn.metrics.accuracy_score(test_Y, test_y_hat)
    print('accuracy_score: {0}'.format(accuracy_score))

    roc_auc_score = sklearn.metrics.roc_auc_score(test_Y, test_y_hat)
    print('roc_auc_score: {0}'.format(roc_auc_score))

    false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(test_Y,test_y_hat)
    #print('Top 10 important features')
    #print(sorted(zip(classifier.feature_importances_, vect.get_feature_names()), reverse=True)[:10]) 
    
    false_pos = []
    false_neg = []
    
    false_pos_probs = []
    false_neg_probs = []

    pos_class_probs = []    

    pos_class_probs_low = []
    pos_class_probs_high = []

    y_hat_proba = classifier.predict_log_proba(test_X)

    for n in range(len(test_ids)):

        norm_proba = [float(j) / sum(y_hat_proba[n]) for j in y_hat_proba[n]]

        if y_hat_proba[n][1] < 0:
            norm_proba[1] = 1 - norm_proba[1]
         
        pos_class_probs.append(norm_proba[1])
    
    pos_class_probs = np.array(pos_class_probs)
    threshold = np.median(pos_class_probs)

    for proba in pos_class_probs:
        if proba >= threshold:
            pos_class_probs_high.append(proba)
        else:
            pos_class_probs_low.append(proba)
    
    pos_class_probs_low = np.array(pos_class_probs_low)
    pos_class_probs_high = np.array(pos_class_probs_high)

    threshold_low = np.median(pos_class_probs_low)
    threshold_high = np.median(pos_class_probs_high)
    
    for n, proba in enumerate(pos_class_probs):    
        if proba <= threshold_low and test_Y[n]==1:
            false_neg.append(test_ids[n])
            false_neg_probs.append(proba)

        elif proba >= threshold_high and test_Y[n]==0:
            false_pos.append(test_ids[n])
            false_pos_probs.append(proba)

    print('Positive class proba distribution')
    #print(pos_class_probs)
    pos_class_probs = np.array(pos_class_probs)
    print('stat')
    print(st.describe(pos_class_probs))
    print('median')
    print(np.median(pos_class_probs))
    print('\n')

    print('False negatives with p of positive class <= %.3f' % threshold_low)
    print('admission ids')
    print(false_neg)
    if len(false_neg_probs)> 0:
        print('p distribution')
        #print(false_neg_probs)
        false_neg_probs = np.array(false_neg_probs)
        print('stat')
        print(st.describe(false_neg_probs))
        print('median')
        print(np.median(false_neg_probs))
        print('\n')


    print('False positives with p of positive class >= %.3f' % threshold_high)
    print('admission ids')
    print(false_pos)
    if len(false_pos_probs)> 0:
        print('p distribution')
        #print(false_pos_probs)
        false_pos_probs = np.array(false_pos_probs)
        print('stat')
        print(st.describe(false_pos_probs))
        print('median')
        print(np.median(false_pos_probs))
        print('\n')

    return fm[1], prec[1], recall[1], roc_auc_score



def extract_featuresBoW(X, val_X, test_X, max_ngram_size=2):

    def make_text_list(X):

        return [' '.join(x) for x in X]

 
    X_texts = make_text_list(X)
    val_X_texts = make_text_list(val_X)
    test_X_texts = make_text_list(test_X)
    all_texts = X_texts + val_X_texts + test_X_texts

    global vect
    vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1, max_ngram_size))
   
    vect.fit(all_texts) 
    X_features = vect.transform(X_texts)
    val_X_features = vect.transform(val_X_texts)
    test_X_features = vect.transform(test_X_texts)

    return vect, X_features, val_X_features, test_X_features

def make_predictionsBoW(X, Y, val_X, val_Y, test_X, test_Y, s, test_ids):
    
    classifier = RandomForestClassifier(class_weight="balanced_subsample", max_depth=2)
 
    total_data = X.shape[0]
 
    classifier.fit(X, Y)
    test_y_hat = classifier.predict(test_X)
    prec, recall, fm, support = precision_recall_fscore_support(test_Y, test_y_hat)
    print('F-measure')
    print(fm)


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    """
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
   
        num_workers = 2  
        downsampling = 1e-3 

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[w for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

      
        # init_sims makes the model much more memory-efficient.
        embedding_model.init_sims(replace=True)
 
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

 
    return embedding_model


def extract_features(X, val_X, test_X, max_ngram_size=2):
 
    def make_text_list(X):
        feat_flat = []
        for x in X:
           indexed_doc = []
           for k in x:
               if k in word2index:
                   indexed_doc.append(word2index[k])
           doc = np.array(pad_sequences([indexed_doc], maxlen=maxlen))
           feat_flat.append(doc[0]) 
        
        return np.array(feat_flat)
    
 
    X_features = make_text_list(X)
    val_X_features = make_text_list(val_X)
    test_X_features = make_text_list(test_X)
    
    vect = None
    return vect, X_features, val_X_features, test_X_features

def make_predictions(X, Y, val_X, val_Y, test_X, test_Y, s, test_ids):
    
    cl_w = compute_class_weight('balanced', np.unique(Y), Y)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience, \
                          verbose=1, mode='auto')
  
    print('Build model CNN model')
  
    in_txt = Input(name='in_norm',
                   batch_shape=tuple([None, maxlen]), dtype='int32')

    # init with pre-trained embeddings
    emb_char = Embedding(len(word2index),
                        embedding_dims,embeddings_initializer=Constant(embedding_matrix),
                           trainable=True, input_length=maxlen, name='emb_char')
 
    emb_seq = emb_char(in_txt)

    z = Dropout(dropout_prob[0])(emb_seq)

    # convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializer_func)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu", kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializer_func)(z)

    out_soft = Dense(1,activation='sigmoid', name='out_soft', kernel_initializer=initializer_func, kernel_regularizer=regularizers.l2(0.01))(z)

    model = Model(inputs=in_txt, outputs=out_soft)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
   
    model.fit(X,Y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(val_X, val_Y), class_weight={0: cl_w[0], 1: cl_w[1]}, callbacks=[earlystop], verbose=0)

    y_hat = model.predict(val_X, batch_size=batch_size)
    y_hat = y_hat.flatten()
    res_list = {}
    thresholds = np.arange(0, 1, 0.1)

    f1_prod_list = []


    for p in thresholds:

        y_pred = []

        for y in y_hat:

            if y >= p:
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_pred = np.array(y_pred)

        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(val_Y, y_pred, average=None)
        f1_prod_list.append(np.prod(f1))
        res_list[p] = y_pred

    f1_prod_list = np.array(f1_prod_list)

    max_f1 = np.argmax(f1_prod_list)
    print('Positive class probability threshold %.4f' % thresholds[max_f1])
    p = thresholds[max_f1]
    y_hat = model.predict(test_X, batch_size=batch_size)
    y_hat = y_hat.flatten()
    y_pred = []

    for y in y_hat:

        if y >= p:
            y_pred.append(1)
        else:
            y_pred.append(0)

    y_pred = np.array(y_pred)
    test_y_hat = y_pred

    prec, recall, fm, support = precision_recall_fscore_support(test_Y, test_y_hat)

    print('F-measure')
    print(fm)
    print('Precision')
    print(prec)
    print('Recall')
    print(recall)
    print('Stat')
    print(support)

    accuracy_score = sklearn.metrics.accuracy_score(test_Y, test_y_hat)
    print('accuracy_score: {0}'.format(accuracy_score))

    roc_auc_score = sklearn.metrics.roc_auc_score(test_Y, test_y_hat)
    print('roc_auc_score: {0}'.format(roc_auc_score))
 
    false_positive_rate, true_positive_rate, thresholds_roc = sklearn.metrics.roc_curve(test_Y, test_y_hat)

    false_pos = []
    false_neg = []

    false_pos_probs = []
    false_neg_probs = []

    pos_class_probs = []

    pos_class_probs_low = []
    pos_class_probs_high = []

    y_hat_proba = y_hat

    pos_class_probs = np.array(y_hat_proba)
    threshold = thresholds[max_f1]

    for proba in pos_class_probs:
        if proba >= threshold:
            pos_class_probs_high.append(proba)
        else:
            pos_class_probs_low.append(proba)

    pos_class_probs_low = np.array(pos_class_probs_low)
    pos_class_probs_high = np.array(pos_class_probs_high)

    threshold_low = np.median(pos_class_probs_low)
    threshold_high = np.median(pos_class_probs_high)

    for n, proba in enumerate(pos_class_probs):
        if proba <= threshold_low and test_Y[n] == 1:
            false_neg.append(test_ids[n])
            false_neg_probs.append(proba)

        elif proba >= threshold_high and test_Y[n] == 0:
            false_pos.append(test_ids[n])
            false_pos_probs.append(proba)

    print('Positive class proba distribution')
    #print(pos_class_probs)
    pos_class_probs = np.array(pos_class_probs)
    print('stat')
    print(st.describe(pos_class_probs))
    print('median')
    print(np.median(pos_class_probs))
    print('\n')

    print('False negatives with p of positive class <= %.3f' % threshold_low)
    print('admission ids')
    print(false_neg)
    if len(false_neg_probs) > 0:
        print('p distribution')
        # print(false_neg_probs)
        false_neg_probs = np.array(false_neg_probs)
        print('stat')
        print(st.describe(false_neg_probs))
        print('median')
        print(np.median(false_neg_probs))
        print('\n')

    print('False positives with p of positive class >= %.3f' % threshold_high)
    print('admission ids')
    print(false_pos)
    if len(false_pos_probs) > 0:
        print('p distribution')
        # print(false_pos_probs)
        false_pos_probs = np.array(false_pos_probs)
        print('stat')
        print(st.describe(false_pos_probs))
        print('median')
        print(np.median(false_pos_probs))
        print('\n')

    return fm[1], prec[1], recall[1], roc_auc_score

conditions = ['F20', #0
              'F32', #1
              'F60', #2
              'F31', #3
              'F25', #4
              'F10'] #5
 

def main():

	all_fms = []
	all_recalls = []
	all_precs = []
	all_aucs = []

	adm_ids = []
	class_labels = []
	text = []
	text_real =[]
	
	# get input data (test data for generation models)
	test_doc_ids = [line.rstrip('\n') for line in open(sys.argv[1])]
	test_texts = [line.rstrip('\n') for line in open(sys.argv[2])]
	test_real_texts = [line.rstrip('\n') for line in open(sys.argv[3])]         
	
	gen_dict = defaultdict(list)
	gen_real_dict = defaultdict(list)
	
	
	for n, doc_id in enumerate(test_doc_ids):

		gen_text = test_texts[n]
		gen_dict[doc_id].extend(gen_text.split(' '))
		gen_real_text = test_real_texts[n]
		gen_real_dict[doc_id].extend(gen_real_text.split(' '))
	
	df = pd.read_csv('input.csv')
	df = df.fillna('')
	
	# get labels from original DB dump
	for row in df.iterrows():
		row = row[1]
		class_row = np.zeros(len(conditions)) 
		doc_id = str(row['Document_ID'])
		if doc_id not in gen_dict.keys():
			continue 
		for n, dia in enumerate(conditions):
			if (row['Diagnosis'] == dia):
				class_row[n] = 1
		adm_ids.append(doc_id)
		class_labels.append(class_row)
	
	for doc_id in adm_ids:
			  
		text.append((doc_id,gen_dict[doc_id]))
		text_real.append((doc_id,gen_real_dict[doc_id]))

	class_labels = np.array(class_labels)

	skf = StratifiedKFold(n_splits=5, random_state=8)

	for trainval_index, test_index in skf.split(text, np.argmax(class_labels, axis=1)):
		
		xtrainval, xtest = [x for i, x in enumerate(text) if i in trainval_index], [x for i, x in enumerate(text_real) if i in test_index]
		ytrainval, ytest = [x for i, x in enumerate(class_labels) if i in trainval_index], [x for i, x in enumerate(class_labels) if i in test_index]
		train_size = int(round(len(text) * .7))
		xtrain, xval, ytrain, yval = train_test_split(xtrainval, ytrainval, train_size=train_size, random_state=8)
		print(len(xtrain))
		print(len(xval))
		print(len(xtest))
		doc_length_array = []
		
		train_x = [x[1] for x in xtrain]
	   
		global word2index
		global embedding_weights

		word2index = {'padding': 0}
		index2word = {0: 'padding'}

		counter = 1
		cc = 0
        
        # define max doc length for padding
		for x in train_x:

			doc_length_array.append(len(x))
			for word in x:
				if word not in word2index:
					word2index[word] = counter
					index2word[counter] = word
					counter += 1
		global maxlen
		maxlen = int(np.percentile(doc_length_array, 75, axis=0))
		print(np.mean(doc_length_array))
		print(maxlen)
		x_train = [line.rstrip('\n').split(' ') for line in open('train.txt')]
		w2v_model = train_word2vec(x_train, index2word, num_features=embedding_dims,
								   min_word_count=1, context=10)
		print('absent')
		print(cc)
		global embedding_matrix
		embedding_matrix = np.zeros((len(word2index), embedding_dims))
		for i in range(len(word2index)):
			if index2word[i] in w2v_model:
				embedding_vector = w2v_model.wv[index2word[i]]
				embedding_matrix[i] = embedding_vector


		print('Computing features')
		valid_x = [x[1] for x in xval]
		test_x = [x[1] for x in xtest]
		
		test_ids = [x[0] for x in xtest]

		vect, X_features, val_X_features, test_X_features = extract_features(train_x, valid_x, test_x, 5)
		ytrain = np.array(ytrain)
		yval = np.array(yval)
		ytest = np.array(ytest)

		print('Training loop')
		for index, condition in enumerate(conditions):
			print('Current Condition: {0}'.format(condition))
			
			train_y = ytrain[:,index]
			valid_y = yval[:,index]
			test_y = ytest[:,index]


			current_fms = []
			current_recalls = []
			current_precs = []
			current_aucs = []
			for y in range(1):
				fm, prec, recall, auc = make_predictions(X_features, train_y, val_X_features, valid_y, test_X_features, test_y, 1, test_ids)  # s)
				current_fms.append(fm)
				current_precs.append(prec)
				current_recalls.append(recall)
				current_aucs.append(auc)
			print('\n')
			current_fms = np.array(current_fms)
			current_precs = np.array(current_precs)
			current_recalls = np.array(current_recalls)
			current_aucs = np.array(current_aucs)
			all_fms.extend(current_fms)
			all_precs.extend(current_precs)
			all_recalls.extend(current_recalls)
			all_aucs.extend(current_aucs)
	print('F-measure all')
	print(all_fms)
	print('Precision all')
	print(all_precs)
	print('Recall all')
	print(all_recalls)
	print('Aucs all')
	print(all_aucs)


if __name__ == "__main__":
    main()
 
