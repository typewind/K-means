import numpy as np
import matplotlib.pyplot as plt
import random
import operator
np.set_printoptions(threshold=np.inf)
# K = number of cluster, 2 <= K <= 20
K=2
# choice=1:instance, choice=2:mean
choice=2
# Path of data
file="Absolute path of train negative here"
# evaluation list
macro_precisions=[]
macro_recalls=[]
macro_f_scores=[]
# tolerance
tolerance=1e-4
# Max Iteration
max_iteration=200
# Debug mode for instance centroid
debug_cen=False

# Generate the dictionary of labels of each review and the whole feature space
def get_featspace(fname):
    labels={}
    featspace={}
    with open(fname) as file:
        for i,line in enumerate(file):
            labels[i]=line.strip().split()[0]
            # genelise feature space
            for word in line.strip().split()[1:]:
                featspace.setdefault(word,1)
            for index, label in enumerate(featspace):
                featspace[label]=index
    return featspace, labels
    
# Map the data into a 408*4861 matrix and do the L2 normalisation  
def get_matrix(fname):
    # 408 lines, 4861 features in total
    matrix=np.zeros((len_labels,len_featspace))
    with open(fname) as file:
        for index, line in enumerate(file):
            review=[]
            for word in line.strip().split()[1:]:
                review.append(featspace[word])
            for i in range(len(review)):
                # L2 Normalisation of reviews
                matrix[index][review[i]]=1/(np.sqrt(len(review)))
    return matrix

# print the different rates from K=2 to K=2O
def print_rates(name,rates):
    print(name,":")
    for index, rate in enumerate(rates):
        print("K =",index+2,"\t",name,"=",rate)
    print("=============")

# class definition of K-means
class K_means:
    def __init__(self, k=K, tol=tolerance, maxIter=max_iteration):
        # k clusters
        self.k=k
        # max itrations
        self.maxIter=maxIter
        # store the vectors of centroids
        self.centroids={}
        # store the information of clusters
        self.clusters={}
        # tolerance
        self.tol=tol
    
    # choose k instances as centroids randomly
    # used in the 1st iteration as an initialisation
    def rand_centroids(self, reviews):
        random=np.random.choice(len_labels,self.k,replace=False)
        for index, i in enumerate(random):
            self.centroids[index]=reviews[i]
    
    # calculate the distance between reviews(instance) and centroid
    def ins_distance(self, review):
        distance=[np.linalg.norm(review-self.centroids[i]) for i in range(len(self.centroids))]
        return distance
    
    # calculate the distance between the current centroid and previous centroid
    def cent_distance(self, pre_centroids):
        distances=[np.linalg.norm(self.centroids[i]-pre_centroids[i]) for i in pre_centroids]
        return distances
    
    # assgin the reviews to the closest centroid corresponding to different cluster
    def clustering(self, reviews):
        self.clusters={}
        for label in self.centroids:
            self.clusters[label]=[]
        for index, review in enumerate(reviews):
            self.clusters[np.argmin(self.ins_distance(review))].append(index)
    
    # calculate the mean centroid    
    def get_centroids_mean(self, reviews):
        for label, cluster in self.clusters.items():
            if(cluster):
                a=np.zeros((1,len_featspace))
                for index in cluster:
                    a+=reviews[index]
                self.centroids[label]=a/len(cluster)
    
    # find the instance which is nearest to the mean centroid
    def get_centroids_instance(self, reviews):
        dis=[]
        for label, cluster in self.clusters.items():
            d=[]  
            for i in range(len(cluster)):
                d.append(np.linalg.norm(reviews[cluster[i]]-self.centroids[label]))
        
            dis.append(d)
        for index, i in enumerate(dis):
            if (i):
                min_index=np.argmin(i)
                self.centroids[index]=reviews[min_index]

    # merge 2 dictionary by sum the value with same key       
    def merge_dict(self,dict_a,dict_b):
        dict_c={}
        for d in (dict_a,dict_b):
            for label, value in d.items():
                if (label in dict_c):
                    dict_c[label]+=value
                else:
                    dict_c[label]=value
        return dict_c
    
    # print the dictionary of evaluation
    def print_eva(self,eva):
        for label in eva:
            print("No.",label+1,"cluster :",eva[label])
    
    # calculate the macro averaged precisions, recalls and f-scores
    def evaluation(self):
        # evaliation consequence
        eva={}
        # the clusters after merge
        merged_clusters={}
        # the label of highest sequence of each cluster
        tags=[]
        # the tags after merge
        merged_tags={}
        # precision of each cluster
        precisions=[]
        # recall of each cluster
        recalls=[]
        # f_score of each cluster
        f_scores=[]
        for label,values in self.clusters.items():
            d={}
            for value in values:
                if(labels[value] in d):
                    d[labels[value]]+=1
                else:
                    d[labels[value]]=1
            eva[label]=d
        print("When K =",self.k,",Clusters are:")
        self.print_eva(eva)
        for label, cluster in eva.items():
            # find the label
            if(cluster):
                tag=max(cluster.items(), key=operator.itemgetter(1))[0]
                tags.append(tag)
        
        # merge the cluster with same label
        for index, tag in enumerate(tags):
            if (tag in merged_tags):
                merged_clusters[merged_tags[tag]]=self.merge_dict(merged_clusters[merged_tags[tag]],eva[index])
            else:
                merged_tags[tag]=index
                merged_clusters[index]=eva[index]
        
        # calculate the precision, recall and f_score of each cluster        
        for label, cluster in merged_clusters.items():
            # if cluster is not empty
            if(cluster):
                precision=max(cluster.values())/sum(cluster.values())
                precisions.append(precision)
                # each label has 51 lines 
                recall=max(cluster.values())/51
                recalls.append(recall)
                # f_score
                f_scores.append(2*precision*recall/(recall+precision))
        # avoid the warning of divide 0
        if(precisions and recalls):
            macro_precisions.append(sum(precisions)/len(precisions))
            macro_recalls.append(sum(recalls)/len(recalls))
            macro_f_scores.append(sum(f_scores)/len(f_scores))
        else:
            # used for debug mode to find which k cannot converging in max iteration
            print(self.k)
    
    # function of iterating the K-means with instance centroid
    def iter_instance(self,reviews):
        self.rand_centroids(reviews)
        pre_centroids=dict(self.centroids)
        self.clustering(reviews)
        # For instance centroid, the algorithm cannot converge to a certain point
        # Thus set the max iteration to 10 to reduce the amount of calculation
        for i in range(10):
            self.get_centroids_mean(reviews)
            self.get_centroids_instance(reviews)
            self.clustering(reviews)
            # Debug mode: calculate the distance between 2 instance centroid
            if (debug_cen==True):
                distances=self.cent_distance(pre_centroids)
                #print("distances:",distances)
                self.evaluation()
                #print(max(distances))
                if(max(distances)< self.tol):
                    #print("converge")
                    self.evaluation()
                    break
                else:
                    pre_centroids=dict(self.centroids)
                    self.clustering(reviews)
        self.evaluation()
        print("============")
    
    #function of iterating the K-means with mean centroid            
    def iter_mean(self,reviews):
        self.rand_centroids(reviews)
        pre_centroids=dict(self.centroids)
        self.clustering(reviews)
        for i in range(self.maxIter):
            self.get_centroids_mean(reviews)
            distances=self.cent_distance(pre_centroids)
            #print("distances:",distances)
            if(max(distances)< self.tol):
                self.evaluation()
                print("Terminated at",i+1,"th iteration")
                print("==========")
                break
            elif(i==self.maxIter-1):
                #print(i)
                self.evaluation()
                print("Terminated in",i+1,"iteration")
                print("==========")
                
            else:
                #self.evaluation()
                pre_centroids=dict(self.centroids)
                self.clustering(reviews)
                
    def iterate(self, reviews, choice):
        if(choice==1):
            self.iter_instance(reviews)
        if(choice==2):
            self.iter_mean(reviews)

                
print("Initialising the feature space...")
featspace, labels = get_featspace(file)
#4861
len_featspace=len(featspace)
#408
len_labels=len(labels)
print("Loading data from file...")
reviews=get_matrix(file)

'''#Calculate the distance between instances
distance_matrix=np.zeros((408,408))
for i in range(len(reviews)):
    for j in range(408):
        distance_matrix[i][j]=np.linalg.norm(reviews[i]-reviews[j])
        if( distance_matrix[i][j]==0 and i!=j):
            print(i,j)
for i in range(408):
    distance_matrix[i].sort()
    print(i,distance_matrix[i][1:6])
'''

# input the choice
print("1.K-means with instance centroids")
print("2.K-means with mean centroids")
choice=int(input("Please input your choice(1 or 2):\n"))

K_axis=[]
# iterate from K=2 to K=20 depending on different choice 
for K in range(2,21):
    K_axis.append(K)
    km=K_means(k=K)
    km.iterate(reviews, choice)
    
# print 3 rates from K=2 to K=20
print_rates("Macro Averaged Precisions",macro_precisions)
print_rates("Macro Averaged Recalls",macro_recalls)
print_rates("Macro Averaged F-Scores",macro_f_scores)

# Draw the plots
plt.figure(figsize=(10,6), dpi=120)
plt.subplot(221)
plt.plot(K_axis,macro_precisions,label="Macro Averaged Precision")
plt.plot(K_axis,macro_recalls,label="Macro Averaged Recall")
plt.plot(K_axis,macro_f_scores,label="Macro Averaged F-Score")
plt.xlabel("K(Number of Clusters)")
plt.xticks(K_axis)
plt.ylabel("Rate")
plt.grid(True)
plt.title('TOL ='+ str(tolerance))
plt.legend()

# Macro Averaged Recall
plt.subplot(222)
plt.plot(K_axis,macro_recalls, label="Macro Averaged Recall", color='C1')
plt.xlabel("K(Number of Clusters)")
plt.ylabel("Rate")
plt.xticks(K_axis)
plt.grid(True)
plt.title("Macro Averaged Recall")

# Macro Averaged Precision
plt.subplot(223)
plt.plot(K_axis,macro_precisions, label="Macro Averaged Precision")
plt.xlabel("K(Number of Clusters)")
plt.ylabel("Rate")
plt.xticks(K_axis)
plt.grid(True)
plt.title("Macro Averaged Precision")

# Macro Averaged F-Score
plt.subplot(224)
plt.plot(K_axis,macro_f_scores, label="Macro Averaged F-Score", color='C2')
plt.xlabel("K(Number of Clusters)")
plt.ylabel("Rate")
plt.xticks(K_axis)
plt.grid(True)
plt.title("Macro Averaged F-Score")

plt.tight_layout()
plt.show()





