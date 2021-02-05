#imports

import nltk
import numpy as np
import xml.etree.cElementTree as ET
import codecs
from nltk.corpus import wordnet
from nltk.wsd import lesk
from string import punctuation
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from google.colab import drive
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection  import train_test_split
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

#Mounting drive
drive.mount('/content/drive')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

#preprocessin definitions
HIGH_PRIORITY_INDEX = 0
lemmatize = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
vectorizer = CountVectorizer()
ps = PorterStemmer()

#class which gives id, lemma, context, index
class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)
#load all the instances dev and test sets
def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    

    dev_instances = {}
    test_instances = {}
    #context,id,lemma and idx simplification
    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            #sentence=simpleFilter(sentence)
            context = [to_ascii(el.attrib['lemma']) for el in sentence]

            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances
#remove stopwords and simplify the sentences
def simpleFilter(sentence):
    filtered_sentence = []

    for w in sentence:
       if w not in stop_words:
          filtered_sentence.append(lemmatize.lemmatize(w))
    return filtered_sentence

#load the keys
def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys.
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue

        if (len(line.strip().split(' ', 2))==3):
           doc, my_id, sense_key = line.strip().split(' ', 2)
        else:
           doc = line.strip().split('.',1)[0]
           my_id, sense_key =  line.strip().split(' ', 2)

        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key
#comvert in to ascii less values
def to_ascii(s):
    s=lemmatize.lemmatize(s)
    return codecs.encode(s, 'ascii', 'ignore')

#Get the wornetsense
def get_sensekey(lemma):
    syns = wordnet.synsets(lemma)
    if len(syns) is not 0:
        return to_sense_key(syns[0])
    else:
        print('Synset is empty')
        return None
#get keys of the sense
def to_sense_key(syn):
    return syn.lemmas()[0].key()

#preporcessing removig stopwords and lemmatize the word
def preProcess(sent):
    sent = [w.lower() for w in sent]
    sent = [w for w in sent if w not in stop_words]
    sent = [lemmatize.lemmatize(w) for w in sent]
    return sent
#Baseline using wordnets to predict
def baseline_wsd(instances, key):

    pred_key = {wsd_id:get_sensekey(word_sense_disamb.lemma.decode("utf-8")) for wsd_id, word_sense_disamb in instances.items()}
    accuracy_score= evaluate_accu(pred_key,key)
    print(accuracy_score)

    return
 
#bootstrapping algorithm to predict senses
def bootstrapping(instances, keys,test_instance,test_keys):
    
    bootstrap_iter = 15
    X=[]
  
    values=[]
    
    idx=[]
    lemmas=[]
    lemmas_test=[]
  
    X_test=[]
    lemmas_total=[]
    X_total=[]
    totalinstances=instances.copy()
   
    totalinstances.update(test_instances)
     #total instances division
    for id, wsd in instances.items():
        idx.append(id)
        lemma = wsd.lemma.decode("utf-8")
        lemma= lemmatize.lemmatize(lemma)
        lemmas.append(get_sensekey(lemma))
        context = [el.decode("utf-8") for el in wsd.context]
        X.append(context)

    for id, wsd in test_instances.items():
        idx.append(id)
        lemma = wsd.lemma.decode("utf-8")
        lemma= lemmatize.lemmatize(lemma)
        lemmas_test.append(get_sensekey(lemma))
        context = [el.decode("utf-8") for el in wsd.context]
        X_test.append(context)

    for id, wsd in totalinstances.items():
       idx.append(id)
       lemma = wsd.lemma.decode("utf-8")
       lemma= lemmatize.lemmatize(lemma)
       lemmas_total.append(get_sensekey(lemma))
       context = [el.decode("utf-8") for el in wsd.context]
       X_total.append(context)


    X=np.asarray(X)
  

    length = max(map(len, X))
    newX_=np.array([xi+[0]*(length-len(xi)) for xi in X])
    enc = OneHotEncoder()
    enc.fit(newX_)
    pos= enc.transform(newX_).toarray()

    #convert in to multilabels
    values= MultiLabelBinarizer().fit_transform(values)
    converted_flattened_string=[]
    converted_flattened_string_test =[]
    converted_flattened_string_total=[]
    #flattening the string
    for x in range(len(X_test)):
       my_lst_str = ' '.join(map(str, X_test[x]))
       converted_flattened_string_test.append(my_lst_str)


    for x in range(len(X)):
       my_lst_str = ' '.join(map(str, X[x]))
       converted_flattened_string.append(my_lst_str)
    for x in range(len(X_total)):
       my_lst_str = ' '.join(map(str, X_total[x]))
       converted_flattened_string_total.append(my_lst_str)
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(converted_flattened_string_total, lemmas_total, test_size = 0.1, random_state = 54)
    accuracy=[]
    accuracy_train=[]
    accuracy_total=[]
    accuracy_train_total=[]
    #bootstrapping 
    for i in range(bootstrap_iter):
        #resampling the data 
        X_,Y_= resample(X_train, y_train)
        #classifier multinobial which classifies data
        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',MultinomialNB(alpha=3.1))])
        #classifying and predicting training and test semse
        classifier.fit(X_,Y_)
        pred=classifier.predict(X_)
        acc_train=accuracy_score(pred,Y_)
        accuracy_train_total.append(acc_train)
        predicted = classifier.predict(X_test)
        accc = accuracy_score(predicted, y_test)
        accuracy_total.append(accc)
    #overall accuracies
    accuracy_total = np.array(accuracy_total)
    accuracy_train_total=np.array(accuracy_train_total)
    predicted_out_total = classifier.predict(X_test)
    accuray_test_total = accuracy_score(predicted_out_total, y_test)
    predicted_train_total = classifier.predict(X_train)
    acc_train_total = accuracy_score(predicted_train_total, y_train)
    print('Accuracy Score_total')
    print('Avearge total: ', accuracy_total.mean())
    print('train avg total',accuracy_train_total.mean())
    print('accuracy score total', accuray_test_total)
    print('train_accuracy score total', acc_train_total)
    #bootstrapping anotjer model
    for i in range(bootstrap_iter):
        X_,Y_= resample(converted_flattened_string, lemmas)
        #classifier multinobial which classifies data
        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',MultinomialNB(alpha=3.1))])
        
        classifier.fit(X_,Y_)
        pred=classifier.predict(X_)
        acc_train=accuracy_score(pred,Y_)
        accuracy_train.append(acc_train)
        predicted = classifier.predict(converted_flattened_string_test)
        acc = accuracy_score(predicted, lemmas_test)
        accuracy.append(acc)
    #overall accuracies
    accuracy = np.array(accuracy)
    accuracy_train=np.array(acc_train)
    predicted_out = classifier.predict(converted_flattened_string_test)
    acc = accuracy_score(predicted_out, lemmas_test)
    predicted_train = classifier.predict(converted_flattened_string)
    acc_train = accuracy_score(predicted_train, lemmas)
    print('Accuracy Score')
    print('Avearge: ', accuracy.mean())
    print('train avg',acc_train.mean())
    print('accuracy score', acc)
    print('train_accuracy score', acc_train)

    return
#accuracy calculation
def evaluate_accu(predict_key,key):
    count= len([id for id, sense in predict_key.items() if id in key and sense in key[id]])
    accuracy = float(count) / \
           len([s for _, s in predict_key.items() if s is not None])
    return accuracy

#using lesk algorithm to predict
def _lesk(context, lemma):
    context= [lemmatize.lemmatize(l).lower() for l in context if l not in punctuation and  stopwords.words("english")]
    #nltk lesk
    synset = lesk(context, lemma.lower())
    if synset is not None:
        sense_key=[sense.key() for sense in synset.lemmas()]
        return sense_key[0]
    else:
        print('synset is empty')
        return None


#modified lesk to predict
def leskmod(context,lemma):
    synsets = wordnet.synsets(lemma)
    syn_defin = [synset.definition() for synset in synsets]
    syn_defin = [preProcess(definition) for definition in syn_defin]
    context = preProcess(context)
    idf, _ = inverFrequency(syn_defin)
    overlap = 0
    synset_index = 0
    
    for syn_index, defin in enumerate(syn_defin):
        if syn_index >= 3:
            pass
        else:
            #calculate the overlap and find the high priority
            current_overlap = getOverlap(context, defin, idf)
            if current_overlap > overlap:
                synset_index = syn_index

    return to_sense_key(synsets[synset_index])

#uses invertfrequency to find the df, idf
def inverFrequency(definition):
    vocabukary = [w for defin in definition for w in defin]
    df = {w:0. for w in vocabukary}
    N = float(len(definition))
    for w in vocabukary:
        for defin in definition:
            if w in defin:
                df[w] += 1.
    idf = {}
    for w, freq in df.items():
        idf[w] = np.log(0.5 + 0.5*N / freq)
    return idf, df

#uses getoverlap function to find the overlap
def getOverlap(context, definition, idf):
    overlap = 0.
    cur_context = context.copy()
    for _ in cur_context:
        w = cur_context.pop()
        try:
            w_index = definition.index(w)
            definition.pop(w_index)
            overlap += idf[w]
        except ValueError:
            pass
    return overlap

#lesk function accuracy calculation
def lesk_wsd(instances, key):

    pred_key = {wsd_id:_lesk([w.decode("utf-8") for w  in word_sense_disamb.context ],word_sense_disamb.lemma.decode("utf-8")) for wsd_id, word_sense_disamb in instances.items()}
    accuracy_score= evaluate_accu(pred_key,key)
    print(accuracy_score)


    return

#lesk modified accuracy calculation
def lesk_modified(instances, key):

    pred_key = {wsd_id:leskmod([w.decode("utf-8") for w  in word_sense_disamb.context ],word_sense_disamb.lemma.decode("utf-8")) for wsd_id, word_sense_disamb in instances.items()}
    accuracy_score= evaluate_accu(pred_key,key)
    print(accuracy_score)


    return

#main function
if __name__ == '__main__':
    data_f = '/content/drive/My Drive/Colab Notebooks/multilingual-all-words.en.xml'
    key_f = '/content/drive/My Drive/Colab Notebooks/wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}

    data_s = '/content/drive/My Drive/Colab Notebooks/semcor.data.xml'
    key_s = '/content/drive/My Drive/Colab Notebooks/semcor.gold.key.txt'
    dev_instances1, test_instances1 = load_instances(data_s)
    devkey1,testkey1 = load_key(key_s)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances1 = {k:v for (k,v) in dev_instances1.items() if k in devkey1}
    test_instances1 = {k:v for (k,v) in test_instances1.items() if k in testkey1}
    # read to use herez = x.copy()

    totatdev_instances= dev_instances.copy()
    totatdev_instances.update (dev_instances1)
    totattest_instances= test_instances.copy()
    totattest_instances.update (test_instances1)


    totatdev_key= dev_key.copy()
    totatdev_key.update (devkey1)
    totattest_key= test_key.copy()
    totattest_key.update (testkey1)

    print(len(totatdev_instances))
    print(len(totattest_instances))

    baseline_wsd(dev_instances,dev_key)
    baseline_wsd(test_instances,test_key)

    lesk_wsd(dev_instances,dev_key)
    lesk_wsd(test_instances,test_key)

    lesk_modified(dev_instances,dev_key)
    lesk_modified(test_instances,test_key)
    bootstrapping(dev_instances,dev_key,test_instances,test_key)
