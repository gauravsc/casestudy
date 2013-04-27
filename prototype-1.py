from gensim import corpora, models
import numpy.random
numpy.random.seed(10)
import random


def get_random_bow(k,total_vocab,max_occ_one_word):
	words_in_doc=random.sample(range(total_vocab),k)
	doc_bow=[]
	for i in range(k):
		doc_bow.append((words_in_doc[i],random.randint(1,max_occ_one_word)))	
	return doc_bow	

def create_corpus_bow(number_of_documents,k,total_vocab,max_occ_one_word):
	corpus=[]
	for i in range(0,number_of_documents):
		doc_bow=get_random_bow(k,total_vocab,max_occ_one_word)
		corpus.append(doc_bow)
	return corpus
	
corpus=create_corpus_bow(100,500,2000,15);

dictionary={}
for i in range(0,2000):
	dictionary[i]=i
	
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lda = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10)
corpus_lda = lda[corpus]

#print lda.print_topics()  
#print lda.expElogbeta
#print sum((lda.expElogbeta)[0])

def get_topic_distribution_doc(top_doc):
	top_doc_list={}
	for  i in range(len(top_doc)):
		top_doc_list[top_doc[i][0]]=top_doc[i][1]
	return top_doc_list
	
num_doc=len(corpus_lda);
doc_vectors=[0]* num_doc;
for i,j in enumerate(corpus_lda):
	doc_vectors[i]=get_topic_distribution_doc(j)


print "\n\n\n\n\n"
print doc_vectors;
	
