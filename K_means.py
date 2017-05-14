import matplotlib.pyplot as plt
#from matplotlib import style
import numpy as np
import numpy.linalg as LA
X=np.array([[2.5,2],[1.7,4],[1.6,6],[2.2,5],[3,1],[3,4],[3.3,2],[2,3.3],[1,2.4]])
#X=np.array([])
class K_means:
    def __init__(self,k=2, tol=1e-4, maxIter=10):
        self.k=k
        self.tol=tol
        self.maxIter=maxIter
    
    def fit(self,data):
        self.centroids={}
        np.random.shuffle(data)
        
        for i in range(self.k):
            self.centroids[i]=data[i]
        #print ("centroids",self.centroids)    
        for i in range(self.maxIter):
            self.classifications={}
            
            for i in range(self.k):
                self.classifications[i]=[]
                
            for featureset in X:
                #print("featureset:",featureset)
                distance=[LA.norm(featureset-self.centroids[centroid],2) for centroid in self.centroids]
                #print("distance:",distance)
                #print("distance=",distance)
                classification=distance.index(min(distance))
                #print("class:",classification)
                #print("classificaiton=",classification)
                self.classifications[classification].append(featureset)
            #print("class 0:",self.classifications[0])
            #print("class 1:",self.classifications[1])
            prev_centroids=dict(self.centroids) 
            for classification in self.classifications:
               # self.centroids[classification]=np.average(self.classifications[classification])
                pass        
            optimised=True
            for c in self.centroids:
                if np.sum((self.centroids[c]-prev_centroids[c])/prev_centroids[c]*100)<self.tol:
                    optimised=False
            if optimised==True:
                break
            

    def predict(self,data):
        distance=[LA.norm(data-self.centroids[centroid],2) for centroid in self.centroids]
        #print("distance=",distance)
        classification=distance.index(min(distance))
        return classification


'''
plt.scatter(X[:,0],X[:,1], color="b",edgecolors="none",label="points")
KM=K_means();
KM.fit(X);
print(KM.classifications)
for centroid in KM.centroids:
    #print("centroid:",centroid)
    #print("centroids[centroid][0]:",KM.centroids[centroid][0])
    plt.scatter(KM.centroids[centroid][0], KM.centroids[centroid][1],color="r")
#print("KM.centroids:",KM.centroids)
for classification in KM.classifications:
    #KM.classifications[classification]=KM.predict(X)
    for featset in classification:
        plt.scatter(KM.classifications[classification][0],KM.classifications[classification][1])
    
#KM.classifications 
#{0: [array([ 3.3,  2. ]), array([ 2.5,  2. ]), array([ 3.,  1.])], 
#1: [array([ 1.7,  4. ]), array([ 3.,  4.]), array([ 2.2,  5. ]), array([ 2. ,  3.3]), array([ 1. ,  2.4]), array([ 1.6,  6. ])]}
   
plt.legend()
plt.show()

#print ("KM.classifications",KM.classifications)
'''

