from gensim import corpora, models
import numpy
numpy.random.seed(10)
import random
import clarans
import sys
import itertools
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
	
	

	
def get_min_lb_dist(cluster_1,cluster_2,pairwise_dist_mat):
	min_dist=sys.maxint
	for i in cluster_1:
		for j in cluster_2:
			if 	pairwise_dist_mat[i][j]<=min_dist:
				min_dist=pairwise_dist_mat[i][j]
				
	return min_dist	
def get_min_ub_dist(cluster_1,cluster_2,pairwise_dist_mat):
	max_dist=-1*sys.maxint
	for i in cluster_1:
		for j in cluster_2:
			if 	pairwise_dist_mat[i][j]>=max_dist:
				max_dist=pairwise_dist_mat[i][j]
				
	return max_dist	
				
def get_cmax_for_super_obj(u,v,w,pmatrix_ub,pmatrix_lb):	
	return max(pmatrix_ub[u][v]-pmatrix_lb[u][w]-pmatrix_lb[v][w],pmatrix_ub[u][w]-pmatrix_lb[w][v]-pmatrix_lb[u][v],
	pmatrix_ub[v][w]-pmatrix_lb[u][w]-pmatrix_lb[u][v])
				
				
def get_cmaxmax_for_super_obj(u,v,w,pmatrix_ub,pmatrix_lb):
	
	return max(pmatrix_ub[u][v]-pmatrix_lb[u][w]-pmatrix_lb[v][w],pmatrix_ub[u][w]-pmatrix_lb[w][v]-pmatrix_lb[u][v],
	pmatrix_ub[v][w]-pmatrix_lb[u][w]-pmatrix_lb[u][v])

def get_cminmax_for_super_obj(u,v,w,pmatrix_ub,pmatrix_lb):
	return min(pmatrix_ub[u][v]-pmatrix_lb[u][w]-pmatrix_lb[v][w],pmatrix_ub[u][w]-pmatrix_lb[w][v]-pmatrix_lb[u][v],
	pmatrix_ub[v][w]-pmatrix_lb[u][w]-pmatrix_lb[u][v])

	
def get_minimum_cmax_for_triplets_in_db():	
	sup_obj=[i for i in range(len(pmatrix_ub))]	
	triplets=itertools.combinations(sup_obj,3)
	min_val=sys.maxint
	for one_trip in triplets:
		if(get_cminmax_for_super_obj(one_trip[0],one_trip[1],one_trip[1],pmatrix_ub,pmatrix_lb)<min_val):
			min_val=get_cminmax_for_super_obj(one_trip[0],one_trip[1],one_trip[1],pmatrix_ub,pmatrix_lb);					
	return min_val	
	
def get_maximum_cmax_for_triplets_in_db():	
	sup_obj=[i for i in range(len(pmatrix_ub))]	
	triplets=itertools.combinations(sup_obj,3)
	max_val=sys.maxint*-1
	for one_trip in triplets:
		if(get_cmaxmax_for_super_obj(one_trip[0],one_trip[1],one_trip[1],pmatrix_ub,pmatrix_lb)>max_val):
			max_val=get_cmaxmax_for_super_obj(one_trip[0],one_trip[1],one_trip[1],pmatrix_ub,pmatrix_lb);			
	return max_val	
		
	
	
def get_cmax_inside_superobject(sup_obj,dist):
	triplets=itertools.combinations(sup_obj,3)
	max_val=sys.maxint*-1
	for i,j in enumerate(triplets):
		temp=max((dist[j[0]][j[1]]-dist[j[0]][j[2]]-dist[j[1]][j[2]]),(dist[j[0]][j[2]]-dist[j[0]][j[1]]-dist[j[2]][j[1]]),(dist[j[1]][j[2]]-dist[j[1]][j[0]]-dist[j[2]][j[0]]));
		if temp>=max_val:
			max_val=temp			
	return max_val;	

def get_cmax_inside_all_superobjects(sup_objs,pairwise_dist_mat):
	cmax_for_so=[0]*len(sup_objs)
	for i in range(len(sup_objs)) :
		cmax_for_so[i]= get_cmax_inside_superobject(sup_objs[i],pairwise_dist_mat)
	return cmax_for_so;
			
				
def get_all_trip_between_gi_gi_1(sup_objects,c,c_next,pmatrix_ub,pmatrix_lb,cmax_for_so):
	triplets=itertools.combinations(sup_objects,3)
	ret_T=[]
	for i,j in enumerate(triplets):
		print i,j
		
		if(get_cmax_for_super_obj(j[0],j[1],j[2],pmatrix_ub,pmatrix_lb)>c and get_cmax_for_super_obj(j[0],j[1],j[2],pmatrix_ub,pmatrix_lb)<=c_next):
			print "there-1"
			un_lab=[k for k in j if label_array[k]==-1]
			cmax_values=[cmax_for_so[l] for l in un_lab]
			print cmax_values
			if(len(un_lab)>0 and max(cmax_values)<c_next and min(cmax_values)>c):
				print "there-2"
				ret_T.append(j)
	return ret_T;	
def get_all_trip_between_gi_gi_1_for_case_2(sup_objects,pmatrix_ub,pmatrix_lb):
	triplets=itertools.combinations(sup_objects,3)
	ret_T=[]
	for i,j in enumerate(triplets):
		if (label_array[j[0]]!=-1 and label_array[j[1]]!=-1 and label_array[j[2]]!=-1 and 
		get_freq_of_grop_i(label_array[j[0]],label_array[j[2]],label_array[j[0]],i)>=2):
			ret_T.append(j)
			
	return ret_T;	
	
def get_freq_of_grop_i(a,b,c,i):
	cnt=0;
	if (a==i):
		cnt+=1
	if (b==i):
		cnt+=1
	if (c==i):
		cnt+=1
	return cnt;
	
	
	
		
def assign_group_labels(pmatrix_ub,pmatrix_lb,cmax_for_gi,i):
	sup_objects=[itemp for itemp in range(len(pmatrix_ub))]
	print sup_objects
	T=get_all_trip_between_gi_gi_1(sup_objects,cmax_for_gi[i-1],cmax_for_gi[i],pmatrix_ub,pmatrix_lb,cmax_for_so)
	print T;
	for one_trip in T:
		for p in one_trip:
			if label_array[p]==-1:
				label_array[p]=i
		if(label_array[one_trip[0]]==i and label_array[one_trip[1]]==label_array[one_trip[2]] and label_array[one_trip[1]]<i 
		and get_cmax_for_super_obj(one_trip[0],one_trip[1],one_trip[2],pmatrix_ub,pmatrix_lb)>cmax_for_gi[label_array[one_trip[1]]]):
			label_array[one_trip[2]]=i;
		if(label_array[one_trip[1]]==i and label_array[one_trip[0]]==label_array[one_trip[2]] and label_array[one_trip[0]]<i 
		and get_cmax_for_super_obj(one_trip[0],one_trip[1],one_trip[2],pmatrix_ub,pmatrix_lb)>cmax_for_gi[label_array[one_trip[0]]]):
			label_array[one_trip[2]]=i;
		if(label_array[one_trip[2]]==i and label_array[one_trip[0]]==label_array[one_trip[1]] and label_array[one_trip[0]]<i 
		and get_cmax_for_super_obj(one_trip[0],one_trip[1],one_trip[2],pmatrix_ub,pmatrix_lb)>cmax_for_gi[label_array[one_trip[0]]]):
			label_array[one_trip[0]]=i;	
	print "here-1"	
	T_2=get_all_trip_between_gi_gi_1_for_case_2(sup_objects,pmatrix_ub,pmatrix_lb)	
	print "here-2"			
	for one_trip in T_2:
		if (get_cmax_for_super_obj(one_trip[0],one_trip[1],one_trip[2],pmatrix_ub,pmatrix_lb)>cmax_for_gi[i]):
			if(label_array[one_trip[0]]==i and label_array[one_trip[1]]==i and label_array[one_trip[2]]!=i):
				label_array[one_trip[0]]=-1
			elif(label_array[one_trip[0]]==i and label_array[one_trip[2]]==i and label_array[one_trip[1]]!=i):
				label_array[one_trip[0]]=-1
			elif(label_array[one_trip[1]]==i and label_array[one_trip[2]]==i and label_array[one_trip[0]]!=i):
				label_array[one_trip[2]]=-1
			else:
				k=random.sample(range(3),1)
				label_array[one_trip[k]]=-1							
	return label_array;		
	
doc_vectors,expElogbeta=create_model()
Q,R = numpy.linalg.qr(expElogbeta);
pairwise_dist_mat = numpy.zeros(shape=(num_doc,num_doc))
for i in range(num_doc): # 0-99
	for j in range(i,num_doc):
		pairwise_dist_mat[i][j]=get_distance_between_2_objects(doc_vectors[i],doc_vectors[j],Q)
		pairwise_dist_mat[j][i]=pairwise_dist_mat[i][j]		

nodes= [i for i in range(num_doc)]	
num_clusters=6
a,b,c=clarans.clarans_basic(nodes,num_clusters,50,250,0.0125,pairwise_dist_mat)
clusters=[]
for i in range(num_clusters):
	temp=[]
	for j in range(num_doc):
		if b[i]==a[j]:
			temp.append(j)
	clusters.append(temp)		
		
					
				
pmatrix_lb=numpy.zeros(shape=(num_clusters,num_clusters))
pmatrix_ub=numpy.zeros(shape=(num_clusters,num_clusters))
for i in range(num_clusters):
	for j in range(i,num_clusters):
	    pmatrix_lb[i][j]=get_min_lb_dist(clusters[i],clusters[j],pairwise_dist_mat);
	    pmatrix_lb[j][i]=pmatrix_lb[i][j];
	    pmatrix_ub[i][j]=get_min_ub_dist(clusters[i],clusters[j],pairwise_dist_mat);
        pmatrix_ub[j][i]= pmatrix_ub[i][j]			
		
		
cmax_for_so=get_cmax_inside_all_superobjects(clusters,pairwise_dist_mat);

num_groups=4
cmax_for_gi=[0]*(num_groups+1)
cmax_for_gi[0]=get_minimum_cmax_for_triplets_in_db()
cmax_for_gi[num_groups]=get_maximum_cmax_for_triplets_in_db()
for i in range(num_groups):
	cmax_for_gi[i+1]=cmax_for_gi[i]+(cmax_for_gi[num_groups]-cmax_for_gi[0])/(num_groups)
	
label_array=[-1]*num_clusters	
sup_objects=[itemp for itemp in range(len(pmatrix_ub))]
	
print cmax_for_gi