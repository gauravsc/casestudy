import pickle
import random
import heapq
f_handle= open('pmatrix_2000.dat','r')
pairwise_dist_mat=pickle.load(f_handle)
f_handle=open('clusters_61.dat','r')
clusters=pickle.load(f_handle)
cmax=[0.20767882436400623, 0.48335764872801251, 0.49103647309201859, 0.49971529745602467, 0.50039412182003074, 0.50607294618403675, 0.50775177054804288, 0.5094305949120489, 0.51210941927605502, 0.51678824364006104, 0.52246706800406717, 0.52314589236807318, 0.52682471673207931, 0.53050354109608533, 0.53118236546009145, 0.53286118982409747, 0.5335400141881036, 0.53521883855210961, 0.56389766291611565, 0.59357648728012169]
label_array=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 16, 12, 20, 20, 1, 13, 20, 7, 20, 20, 20, 3, 15, 17, 14, 13, 9, 11, 8, 10, 5, 18, 20, 2, 19, 20, 4, 19, 6, 19, 19, 19, 20, 20, 19]

groups=[[] for i in range(20)]
for i in range(61):
	groups[label_array[i]-1]+=clusters[i]
	
pivot=[0 for i in range(20)]
for i in range(20):
	pivot[i]=random.choice(groups[i])
pivot=[1142, 278, 1403, 588, 590, 910, 335, 545, 522, 445, 870, 986, 1533, 2, 1620, 68, 1175, 1835, 938, 1494]

pivot_dist=[[] for i in range(20)]	
for i in range(20):
	for j in groups[i]:
		dist=pairwise_dist_mat[j,pivot[i]]
		pivot_dist[i].append(dist)
	
	
num_query=100		
query_array=random.sample(range(2000),num_query)
#query_array=[1222]
prune_count=0	
query=query_array[0]
#initialize the top_k elements
top_k=[]
for i in range(10):
		
		dist=pairwise_dist_mat[query,groups[random.randint(0,19)][i]]	
		heapq.heappush(top_k,-dist)

prev_top_k=list(top_k)
print prev_top_k
top_k=list(prev_top_k)
for cnt in range(num_query):	
	query=query_array[cnt]
	top_k=list(prev_top_k)
	#query=381
	#top_k=[]
	#for i in range(10):
	#	dist=pairwise_dist_mat[query,groups[0][i]]	
	#	heapq.heappush(top_k,-dist)
	max_dist=-1*heapq.heappop(top_k)
	for i in range(20):
		for j in range(len(groups[i])):
			if abs(pairwise_dist_mat[query,pivot[i]]-pivot_dist[i][j])-cmax[i]>max_dist:
				prune_count+=1
			else:
				dist=pairwise_dist_mat[query,groups[i][j]]
				if dist<max_dist:
					heapq.heappush(top_k,-dist)	
					max_dist=-1*heapq.heappop(top_k)

print prune_count/num_query
########################################
top_k=list(prev_top_k)

prune_count=0	
for cnt in range(num_query):
	query=query_array[cnt]	
	top_k=list(prev_top_k)
	temp= [i for i in range(20)]
	random.shuffle(temp)

	#top_k=[]
	#for i in range(10):
	#	dist=pairwise_dist_mat[query,groups[temp[0]][i]]	
	#	heapq.heappush(top_k,-dist)

	max_dist=-1*heapq.heappop(top_k)
	

	for i in temp:
		for j in range(len(groups[i])):
			if abs(pairwise_dist_mat[query,pivot[i]]-pivot_dist[i][j])-cmax[i]>max_dist:
				prune_count+=1
			else:
				dist=pairwise_dist_mat[query,groups[i][j]]
				if dist<max_dist:
					heapq.heappush(top_k,-dist)	
					max_dist=-1*heapq.heappop(top_k)

print prune_count/num_query


###########################################
top_k=list(prev_top_k)
prune_count=0
for cnt in range(num_query):
	top_k=list(prev_top_k)
	query=query_array[cnt]	
	temp= [i for i in range(20)]

	temp.reverse()
	#top_k=[]
	#for i in range(10):
	#	dist=pairwise_dist_mat[query,groups[temp[0]][i]]	
	#	heapq.heappush(top_k,-dist)

	max_dist=-1*heapq.heappop(top_k)

	for i in temp:
		for j in range(len(groups[i])):
			if abs(pairwise_dist_mat[query,pivot[i]]-pivot_dist[i][j])-cmax[i]>max_dist:
				prune_count+=1
			else:
				dist=pairwise_dist_mat[query,groups[i][j]]
				if dist<max_dist:
					heapq.heappush(top_k,-dist)	
					max_dist=-1*heapq.heappop(top_k)

print prune_count/num_query

########## improved pivot selection
top_k=list(prev_top_k)
#print top_k
def sum_elements(pairwise_dist_mat,k,elements):
	sum=0
	for i in elements:
		sum+=pairwise_dist_mat[i][k]
		
	return sum
	
prune_count=0
for cnt in range(num_query):
	query=query_array[cnt]	
	top_k=list(prev_top_k)

	temp= [i for i in range(20)]
	temp.reverse()
	for i in range(20):
		p_candidate=groups[i]
		#sums=[pairwise_dist_mat[k].sum() for k in p_candidate] 
		sums=[sum_elements(pairwise_dist_mat,k,groups[i]) for k in p_candidate]
		pivot[i]=p_candidate[sums.index(max(sums))]
	pivot[0]=1142			
	pivot_dist=[[] for i in range(20)]
	for i in range(20):
		for j in groups[i]:
			dist=pairwise_dist_mat[j,pivot[i]]
			pivot_dist[i].append(dist)
		
	
	#query=(random.sample(range(2000),1))[0]
	#query=381
	#top_k=[]
	#for i in range(10):
	#	dist=pairwise_dist_mat[query,groups[0][i]]	
	#	heapq.heappush(top_k,-dist)

	max_dist=-1*heapq.heappop(top_k)
	#print max_dist

	for i in temp:
		for j in range(len(groups[i])):
			#print abs(pairwise_dist_mat[query,pivot[i]]-pivot_dist[i][j])-cmax[i]
			if abs(pairwise_dist_mat[query,pivot[i]]-pivot_dist[i][j])-cmax[i]>max_dist:
				prune_count+=1
			else:
				dist=pairwise_dist_mat[query,groups[i][j]]
				if dist<max_dist:
					heapq.heappush(top_k,-dist)	
					max_dist=-1*heapq.heappop(top_k)

print prune_count/num_query
			
			
			
########## improved pivot selection, minimum value of distances from pivot
prune_count=0
for cnt in range(num_query):
	top_k=list(prev_top_k)
	temp= [i for i in range(20)]
	temp.reverse()
	query=query_array[cnt]		
	for i in temp:
		p_candidate=groups[i]
		#sums=[sum_elements(pairwise_dist_mat,k,groups[i]) for k in p_candidate]
		sums=[pairwise_dist_mat[k].sum() for k in p_candidate] 
		pivot[i]=p_candidate[sums.index(min(sums))]
	pivot[0]=1142		
	pivot_dist=[[] for i in range(20)]
	for i in range(20):
		for j in groups[i]:
			dist=pairwise_dist_mat[j,pivot[i]]
			pivot_dist[i].append(dist)
		
		
	#query=(random.sample(range(2000),1))[0]
	top_k=[]
	for i in range(10):
		dist=pairwise_dist_mat[query,groups[0][i]]	
		heapq.heappush(top_k,-dist)
	top_k=list(prev_top_k)
	max_dist=-1*heapq.heappop(top_k)	


	for i in temp:
		for j in range(len(groups[i])):
			if abs(pairwise_dist_mat[query,pivot[i]]-pivot_dist[i][j])-cmax[i]>max_dist:
				prune_count+=1
			else:
				dist=pairwise_dist_mat[query,groups[i][j]]
				if dist<max_dist:
					heapq.heappush(top_k,-dist)	
					max_dist=-1*heapq.heappop(top_k)

print prune_count/num_query





			