import numpy
import random
import clarans
import sys
import itertools
num_doc=1000
num_topics=20
size_vocab=250
num_clusters=100
num_groups=20



def normalize_vector(vector):
	norm= pow(numpy.dot(vector,vector),0.5)
	temp=[i/norm for i in vector]
	return (temp)

def genrate_vector_representation_for_one_document(num_topics,size_vocab):
	doc_vector=[]
	topic_word= numpy.zeros(shape=(size_vocab,num_topics))
	for i in range(size_vocab):
		for j in range(num_topics):
				topic_word[i][j]=random.randint(0, 1000)
	topic_word /=  topic_word.sum(axis=1)[:,numpy.newaxis]
	U, s, V = numpy.linalg.svd(topic_word, full_matrices=False)
	average_lambda=sum(s)/len(s)
	for i in range(len(s)):
		if(s[i]<average_lambda):
			s[i]=0
	average_lambda_square=pow(sum([pow(s[i],2) for i in range(len(s))]),0.5)
	t=[k/average_lambda_square for k in s]
	topic_wordT=topic_word.T
	for i in range(len(t)):
		if (t[i]>0):
			doc_vector.append((t[i],normalize_vector(topic_wordT[i])))
	return doc_vector
	
def distance_between_two_documents(doc_vector_1,doc_vector_2):
	sum=0
	for i in doc_vector_1:
		for j in doc_vector_2:
		   # ###print"\n\n\n\n\n"
		   # ###print i,j
		    sum=sum+i[0]*j[0]*pow(numpy.dot(i[1],j[1]),2)
	return sum		

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
	return max(0,pmatrix_ub[u][v]-pmatrix_lb[u][w]-pmatrix_lb[v][w],pmatrix_ub[u][w]-pmatrix_lb[w][v]-pmatrix_lb[u][v],
	pmatrix_ub[v][w]-pmatrix_lb[u][w]-pmatrix_lb[u][v])
				
				
def get_cmaxmax_for_super_obj(u,v,w,pmatrix_ub,pmatrix_lb):
	
	return max(pmatrix_ub[u][v]-pmatrix_lb[u][w]-pmatrix_lb[v][w],pmatrix_ub[u][w]-pmatrix_lb[w][v]-pmatrix_lb[u][v],
	pmatrix_ub[v][w]-pmatrix_lb[u][w]-pmatrix_lb[u][v])

def get_cminmin_for_super_obj(u,v,w,pmatrix_ub,pmatrix_lb):
	return min(pmatrix_lb[u][v]-pmatrix_ub[u][w]-pmatrix_ub[v][w],pmatrix_lb[u][w]-pmatrix_ub[w][v]-pmatrix_ub[u][v],
	pmatrix_lb[v][w]-pmatrix_ub[u][w]-pmatrix_ub[u][v])

	
def get_minimum_cmax_for_triplets_in_db():	
	sup_obj=[i for i in range(len(pmatrix_ub))]	
	triplets=itertools.combinations(sup_obj,3)
	min_val=sys.maxint
	for one_trip in triplets:
		if(get_cminmin_for_super_obj(one_trip[0],one_trip[1],one_trip[1],pmatrix_ub,pmatrix_lb)<min_val):
			min_val=get_cminmin_for_super_obj(one_trip[0],one_trip[1],one_trip[1],pmatrix_ub,pmatrix_lb);					
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
		###print i,j
		###print "hello"
		###print get_cmax_for_super_obj(j[0],j[1],j[2],pmatrix_ub,pmatrix_lb);
		if(get_cmax_for_super_obj(j[0],j[1],j[2],pmatrix_ub,pmatrix_lb)>c and get_cmax_for_super_obj(j[0],j[1],j[2],pmatrix_ub,pmatrix_lb)<=c_next):
			###print "there-1"
			un_lab=[k for k in j if label_array[k]==-1]
			cmax_values=[cmax_for_so[l] for l in un_lab]
			###print cmax_values
			#if(len(un_lab)>0 and max(cmax_values)<c_next and min(cmax_values)>c):
			if(len(un_lab)>0 and max(cmax_values)<c_next):
				###print "there-2"
				ret_T.append(j)
	return ret_T;	
	
def get_all_trip_between_gi_gi_1_for_case_2(sup_objects,pmatrix_ub,pmatrix_lb,i):
	triplets=itertools.combinations(sup_objects,3)
	ret_T=[]
	for k,j in enumerate(triplets):
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
	
	
	
		
def assign_group_labels(pmatrix_ub,pmatrix_lb,cmax_for_gi,i,label_array):
	sup_objects=[itemp for itemp in range(len(pmatrix_ub))]
	###print sup_objects
	T=get_all_trip_between_gi_gi_1(sup_objects,cmax_for_gi[i-1],cmax_for_gi[i],pmatrix_ub,pmatrix_lb,cmax_for_so)
	#print T;
	for one_trip in T:
		#print one_trip
		for p in one_trip:
			#print p
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
	print label_array		
	###print "here-1"	
	T_2=get_all_trip_between_gi_gi_1_for_case_2(sup_objects,pmatrix_ub,pmatrix_lb,i)	
	###print "here-2"
	###print T_2			
	for one_trip in T_2:
		if (get_cmax_for_super_obj(one_trip[0],one_trip[1],one_trip[2],pmatrix_ub,pmatrix_lb)>cmax_for_gi[i]):
			if(label_array[one_trip[0]]==i and label_array[one_trip[1]]==i and label_array[one_trip[2]]!=i):
				label_array[random.choice([1,0])]=-1
			elif(label_array[one_trip[0]]==i and label_array[one_trip[2]]==i and label_array[one_trip[1]]!=i):
				label_array[random.choice([0,2])]=-1
			elif(label_array[one_trip[1]]==i and label_array[one_trip[2]]==i and label_array[one_trip[0]]!=i):
				label_array[random.choice([1,2])]=-1
			elif(label_array[one_trip[1]]==i and label_array[one_trip[2]]==i and label_array[one_trip[0]]==i):
				k=random.sample(range(3),1)				
				label_array[one_trip[k[0]]]=-1							
	return label_array;		
		
	


	
print "starting document generation"	
doc_vectors=[]
for i in range(num_doc):
	####print '%d/500\n' %i 
	doc_vector=genrate_vector_representation_for_one_document(random.randint(20,25),size_vocab)
	doc_vectors.append(doc_vector)
	
print "starting pairwise matrix generation"		
pairwise_dist_mat = numpy.zeros(shape=(num_doc,num_doc))
for i in range(num_doc):
	####print i
	for j in range(i):
		pairwise_dist_mat[i][j]=abs(1-distance_between_two_documents(doc_vectors[i],doc_vectors[j]))
		pairwise_dist_mat[j][i]=pairwise_dist_mat[i][j]
		
			
nodes= [i for i in range(num_doc)]	

#a,b,c=clarans.clarans_itp(nodes,num_clusters,5,250,0.000125,pairwise_dist_mat)
a,b,c=clarans.clarans_basic(nodes,num_clusters,5,250,0.000125,pairwise_dist_mat)
#a,b,c=clarans.clarans_basic(nodes,num_clusters,10,100,0.00125,pairwise_dist_mat)
clusters=[]
for i in range(num_clusters):
	temp=[]
	for j in range(num_doc):
		if b[i]==a[j]:
			
			temp.append(j)
	clusters.append(temp)		
		
temp_list=[]
index_to_delete=[]
for i in range(len(clusters)):
	if len(clusters[i])<3:
		temp_list+=clusters[i]
		index_to_delete.append(i)
		
relative=0		
for i in index_to_delete:		
	del clusters[i-relative]
	relative+=1
	
						
#for i in temp_list:
#	clusters[random.randint(0,len(clusters)-1)].append(i)

clusters.append(temp_list)
	
num_clusters=len(clusters)						
				
pmatrix_lb=numpy.zeros(shape=(num_clusters,num_clusters))
pmatrix_ub=numpy.zeros(shape=(num_clusters,num_clusters))
for i in range(num_clusters):
	for j in range(num_clusters):
	    pmatrix_lb[i][j]=get_min_lb_dist(clusters[i],clusters[j],pairwise_dist_mat);
	    pmatrix_lb[j][i]=pmatrix_lb[i][j];
	    pmatrix_ub[i][j]=get_min_ub_dist(clusters[i],clusters[j],pairwise_dist_mat);
        pmatrix_ub[j][i]= pmatrix_ub[i][j]			
		

cmax_for_so=get_cmax_inside_all_superobjects(clusters,pairwise_dist_mat);
#num_groups=20

cmax_for_gi=[0]*(num_groups+1)
#cmax_for_gi[0]=get_minimum_cmax_for_triplets_in_db()
cmax_for_gi[0]=0
cmax_for_gi[num_groups]=get_maximum_cmax_for_triplets_in_db()
for i in range(num_groups):
	cmax_for_gi[i+1]=cmax_for_gi[i]+(cmax_for_gi[num_groups]-cmax_for_gi[0])/(num_groups)	

label_array=[-1]*num_clusters;	
sup_objects=[itemp for itemp in range(len(pmatrix_ub))]
####print cmax_for_gi
i=1
cnt=0
while(i<=num_groups and cnt <1000):
	###print label_array
	cnt+=1
	prev_label_array=list(label_array)
	label_array=assign_group_labels(pmatrix_ub,pmatrix_lb,cmax_for_gi,i,label_array)
	if(label_array.count(i)<=num_clusters/num_groups and label_array.count(-1)!=0): 
		label_array=list(prev_label_array)
		cmax_for_gi[i]+=0.001
	else:
		cnt=0
		i+=1
		###print "here is the i"
		print i

###print label_array
