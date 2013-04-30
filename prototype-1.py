from gensim import corpora, models
import numpy
numpy.random.seed(10)
import random
num_doc=100

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
	
	
def create_model():	
	corpus=create_corpus_bow(100,500,2000,15);
	dictionary={}
	for i in range(0,2000):
		dictionary[i]=i
	
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	lda = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10	)
	corpus_lda = lda[corpus]
	num_doc=len(corpus_lda);
	doc_vectors=[0]* num_doc;

	for i,j in enumerate(corpus_lda):
		doc_vectors[i]=get_topic_distribution_doc(j)
		
	return (doc_vectors,lda.expElogbeta);

#function is to get topic distribution for each document
def get_topic_distribution_doc(top_doc):
	top_doc_list={}
	for  i in range(len(top_doc)):
		top_doc_list[top_doc[i][0]]=top_doc[i][1]
	return top_doc_list

def get_distance_between_2_objects(doc_vector_1,doc_vector_2,expElogbeta):
	keys_1=doc_vector_1.keys();
	keys_2=doc_vector_2.keys();
	dist=0;
	for i in keys_1:
		for j in keys_2:
			dist+=(doc_vector_1[i]*doc_vector_2[j]*(numpy.dot(expElogbeta[i],expElogbeta[j])))
	return (1-dist);
	
	
doc_vectors,expElogbeta=create_model()
Q,R = numpy.linalg.qr(expElogbeta);
pairwise_dist_mat = numpy.zeros(shape=(num_doc,num_doc))
for i in range(num_doc): # 0-99
	for j in range(i,num_doc):
		pairwise_dist_mat[i][j]=get_distance_between_2_objects(doc_vectors[i],doc_vectors[j],Q)
		pairwise_dist_mat[j][i]=pairwise_dist_mat[i][j]		
	

