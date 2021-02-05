# NLP_Word_sense_Disambiguation-WSD

This projects aims to find the Word Sense Disambiguation (WSD) in Natural Language Processing. Given
instance and keys as defined by a dictionary where each instance dictionary has to classify the word’s
occurrence in context into one or more of its sense classes. I have classified WSD by using Baseline, Lesk
algorithm, bootstrapping, and modified lesk algorithm (Vector space model). I have divided the given data
into (dev_instances,dev_key) as a training dataset of length 194 and (test_instances, test_key) as a test
dataset of size 1450. Each instance is in the form of a dictionary with context and lemma, and I have used
it to predict the sense. I have used the NLTK interface and Wordnet 3.0. I have also preprocessed every
sentence (word-tokenized) and word by Lemmatization and removed the stopwords.

## Methods Used
# Baseline
The Baseline is an algorithm used to classify the data with the help of NLTK wordnet. I have found
each lemma’s synsets from the instance and predicted the top priority sense key for each lemma. The
predicted-key and test-key give the accuracy score of the baseline algorithm. The training accuracy is
51.7 %, and the testing accuracy is 49.5 %. There is a reduction in accuracy by almost 1-2 % when the
stopwords removal and Lemmatization do not occur. I have calculated accuracy manually without using
any built-in libraries.

# Lesk Algorithm
The Lesk algorithm is the heuristic algorithm used to figure out the word sense in the given context. I
have used the built-in NLTK lesk function. I have passed the lemmatized context and lemma to the lesk
function that given a list of synsets. The top prioritized synsets sense key is the predicted key. I have
calculated accuracy against predicted-key and test-key manually. The results turn out to be training
accuracy as 29 % and testing accuracy as 28 %. The results are seemingly shocking on stopwords removal
in the context, which gives accuracy almost 7 % less than the original

# Bootstrapping
Bootstrapping is a Yarowsky’sAlgorithm to predict the sense from the lemmas’ given context with the
supervised model’s help. I have resampled the context and sense of lemmas into a supervised model
(MultinomialNB) 15 times and calculated each iteration’s accuracy. I have calculated the overall accuracy
by taking the mean of these accuracy scores. Since I have already fitted the model with the 15 iterations,
I have calculated the overall testing accuracy using the same model. There are many observations in this
model. The general training (10 %) and testing accuracy(4 %) are comparatively significantly less

# Modified Lesk
The Modified lesk algorithm is similar to the NLTK lesk, except there is little preprocessing and
overlapping. As in the case of the original Lesk, I have taken synsets of each lemma using wordnet. I have
used the vector space model to calculate document frequency and inverse document frequency (df,idf).
I have used the synset’s idf, context, and definition to calculate each synset’s overlap. The key to the
highest overlap value of a synset is the predicted-key. I have figured the accuracy of predicted-key and
test-key manually. The accuracy is much better than the original Lest, with training accuracy of 48 % and
testing accuracy of 47 %. There is an increase in accuracy by almost 1-2 % when the stopwords removal
and Lemmatization do not occur.

# Analysis

1) I have observed from the table that the best model to work during Word base disambiguate is Baseline
as it gave the highest performance among all the algorithms
2) Apart from Lesk and modified-lesk, preprocessing plays an essential role in all the models. In Lesk and
modified-lesk, the algorithm works well with stopwords and without Lemmatization.
3)Though Lesk and Baseline make use of NLTK and Wordnet3.0, the baseline model gives better
performance.
4) The overall performance of all the models is significantly less because the data set is minimal. There
should be at least 80 % of the training data to train a model. But here, training data is substantially less
compared to the testing data and so low performance.
5) Usually, bootstrapping gives better results, and multinomial naive Bayes performs at least 60 % in
classification problems. But here, due to the presence of less data, the performance is significantly less. As
I added new data set from the given website ’https://sites.google.com/site/wsdevaluation/training-data’
of length 1050 to the training dataset, the performance increases to 12 % of overall testing accuracy.
There is a possibility of an increase in performance as data increases.
6)I have also tried bootstrapping with five lexical items (’year,’ ’week,’ ’voluntary,’ ’group,” plan’), which
performs relatively better than the overall bootstrapping, giving 15 % training accuracy and 10 % testing
accuracy. The words which are not in the training set give less performance of the overall testing accuracy.
Therefore, I overlapped and chose the lemmas present in both dev sets and test sets and performed
classification using supervised models, giving a 2 % increase in overall testing-accuracy.

# Results 
## Accuracies
|Algorithms|Training Accuracy| Testing Accuracy with preprocessing (%) | Testing Accuracy without lemmatization (%)| Testing Accuracy with stopwords (%)|
|-----|------|-----|------|------|
Baseline |51.73 |50.72 |49.63| 48.74|
Lesk |21.83 |21.94 |28.3| 29.7|
Bootstrapping |10.73 (overall)<br> 8.7 (average)|8.3 (overall)<br>5.3(average)|7.3| 6.7|
Modified Lesk| 48.89| 47.57 |49.89| 49.96|

## Sample output
Algorithms| Lemmas |Predicted Output
|-----|-----|------|
Baseline |’impact’ <br>’united_states|’impact %1:11:00::’ <br>’united_states %1:15:00::’|
Lesk |’impact’<br>’united_states|’shock %1:04:01::’ <br>’united_states %1:15:00::’|
Bootstrapping |’impact’<br>’united_states|’climate %1:26:00::’ <br>’state %1:14:00::’|
Modified Lesk| ’impact’<br>’united_states|’impact %1:11:00::’<br>’united_states %1:15:00::’|


# Languages 
Python  (NLP word embeddings)

# References 
1) https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.259.147&rep=rep1&type=pdf#::text=idf,similarity%20between%20two%20t%20dimensional
2) https://github.com/maneshreyash/Word−Sense−Disambiguation/blob/master/code.py
3) https://github.com/JGuymont/lesk−algorithm
4) https://github.com/hardik−vala/mylesk/blob/master/lesk.py
5) https://github.com/buomsoo−kim/Machine−learning−toolkits−with−python/blob/master/Model%20selection%20(cross−validation%20vs%20bootstrap)/source%20code/bootstrapping
