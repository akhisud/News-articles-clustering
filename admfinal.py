import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.text import TextCollection
from nltk import translate
from string import punctuation
from random import randrange
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
from collections import namedtuple
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pdb 

'''
Code for corpus generation and preprocessing

'''

stop = stopwords.words('english')
stemmer = PorterStemmer() # possible improvement: more aggressive stemmer
wordnet_lemmatizer = WordNetLemmatizer()
mypath = './toi'



corpus = {}
docs_vs_number={}
count=0

for f in os.listdir(mypath):    
    if (not f.startswith('.')) & os.path.isfile(os.path.join(mypath, f)):    
        with open(os.path.join(mypath, f), 'r') as doc:
            corpus[f] = ''

            #tokenizes words, removes stop words, stems other words
            for i in doc.read().lower().translate(None, punctuation).split():
                if i not in stop:
                    corpus[f] = corpus[f] + ' ' + wordnet_lemmatizer.lemmatize(i)
            if corpus[f]=='':
                del corpus[f]
                continue
            docs_vs_number[f]= count
            count=count+1
            num_docs = count            


'''
Code to generate tf-idf
'''
tfidf = TfidfVectorizer()
tfs = tfidf.fit_transform(corpus.values())


feature_names = tfidf.get_feature_names() 
docs = np.array(corpus.keys())
dense = tfs.todense()


document= dense[0].tolist()[0]
phrase_scores = [pair for pair in zip(range(0, len(document)), document) if pair[1] > 0]


print "In doc matrix: (docs, terms) :", dense.shape

cluster_centres = []
fitness=0
hmcr=0.9
max_imp= 10
hms=5

harmony_memory = []
par = 0.3
par1 = 0.6*par
par2 = 0.3*par
NUM_CLUSTERS=5


"""
    Following functions are needed for the main Harmony Search algorithm. It initializes the harmony memory and then continually generates new harmonies until the stopping criterion (max_imp iterations) is reached.
"""


def find_centre(list_of_docs,dense):
    centre = [0]*len(dense[docs_vs_number[list_of_docs[0]]].tolist()[0])
    
    for doc in list_of_docs:
        centre= [sum(n) for n in zip(*[centre, (dense[docs_vs_number[doc]].tolist()[0]) ] ) ]

    for i in range(0,len(list_of_docs)-1):
        centre[i]=centre[i]/len(list_of_docs)

    return centre

def find_average_cluster_distance(centre,list_of_docs,dense):
    distance = 0
    for doc in list_of_docs:
        distance= distance + cosine_similarity(dense[docs_vs_number[doc]].tolist()[0] , centre)
    distance= distance/len(list_of_docs)
    return distance

def get_fitness(vector,dense):
    """
    Return the objective function value given a solution vector containing each decision variable. In practice,
    vector should be a list of parameters.

    For example, suppose the objective function is (-(x^2 + (y+1)^2) + 4). A possible call to fitness may look like this:

    >>> print obj_fun.fitness([4, 7])
    -76
    """
    
    distance_array= [0]*NUM_CLUSTERS
    cluster_centres=[0]*NUM_CLUSTERS
    docs_in_cluster=[]
    for i in range(0,NUM_CLUSTERS):
        docs_in_cluster.append([])

    for doc in vector.keys():
        docs_in_cluster[vector[doc]].append(doc)

    for cluster in range(0,NUM_CLUSTERS):
        cluster_centres[cluster]= find_centre(docs_in_cluster[cluster],dense)
        distance_array[cluster]=find_average_cluster_distance(cluster_centres[cluster],docs_in_cluster[cluster],dense)
    fitness=sum(distance_array)/NUM_CLUSTERS
    return fitness, cluster_centres

def get_fitness_DB(vector,dense):
    distance_array= [0]*NUM_CLUSTERS
    cluster_centres=[0]*NUM_CLUSTERS
    docs_in_cluster=[]
    for i in range(0,NUM_CLUSTERS):
        docs_in_cluster.append([])

    for doc in vector.keys():
        docs_in_cluster[vector[doc]].append(doc)

    for cluster in range(0,NUM_CLUSTERS):
        cluster_centres[cluster]= find_centre(docs_in_cluster[cluster],dense)
        distance_array[cluster]=find_average_cluster_distance(cluster_centres[cluster],docs_in_cluster[cluster],dense)
    DB = 0 
    for cluster1 in range(0,NUM_CLUSTERS):
        max=0 
        for cluster2 in range(0,NUM_CLUSTERS):
            if cluster1==cluster2:
                continue
            if((distance_array[cluster1]+ distance_array[cluster2])/cosine_similarity(cluster_centres[cluster1],cluster_centres[cluster2])[0][0])>max:
                max = ((distance_array[cluster1]+ distance_array[cluster2])/cosine_similarity(cluster_centres[cluster1],cluster_centres[cluster2])[0][0])
        DB=DB+max

    fitness=DB/NUM_CLUSTERS        
    print "fitness" ,fitness[0][0]
    return (1/fitness[0][0]), cluster_centres

def pitch_adjustment(harmony, doc, cluster_centres,dense):
    """
        If variable, randomly adjust the pitch up or down by some amount. This is the only place in the algorithm where there
        is an explicit difference between continuous and discrete variables.

        The probability of adjusting the pitch either up or down is fixed at 0.5. The maximum pitch adjustment proportion (mpap)
        and maximum pitch adjustment index (mpai) determine the maximum amount the pitch may change for continuous and discrete
        variables, respectively.

        For example, suppose that it is decided via coin flip that the pitch will be adjusted down. Also suppose that mpap is set to 0.25.
        This means that the maximum value the pitch can be dropped will be 25% of the difference between the lower bound and the current
        pitch. mpai functions similarly, only it relies on indices of the possible values instead.
    """
        

    if random.random() < par1:
        
        min=0
        centre_least=-1
        for centre in cluster_centres:
            if(cosine_similarity(dense[docs_vs_number[doc]].tolist()[0] , centre))<min:
                min= cosine_similarity(dense[docs_vs_number[doc]].tolist()[0] , centre)
                centre_least=cluster_centres.index(centre)
        harmony[doc]=centre_least

    if random.random() < par2:
        max_distance=float('-inf')
        sum_distances=0
        probability_centres=[]

        for centre in cluster_centres:
            sum_distances= sum_distances + cosine_similarity(dense[docs_vs_number[doc]].tolist()[0] , centre)[0][0]
            if(cosine_similarity(dense[docs_vs_number[doc]].tolist()[0] , centre)[0][0])>max_distance:
                max_distance= cosine_similarity(dense[docs_vs_number[doc]].tolist()[0] , centre)[0][0]

        for centre in cluster_centres:
            probability=((max_distance-(cosine_similarity(dense[docs_vs_number[doc]].tolist()[0] , centre)[0][0]))/((NUM_CLUSTERS*max_distance)-sum_distances))* (1-(num_imp/max_imp))
            probability_centres.append(probability)

        harmony[doc]= (np.random.choice(NUM_CLUSTERS, 1, p=probability_centres))[0]

    return harmony[doc]
            
def update_harmony_memory(considered_harmony, considered_fitness,harmony_memory):
    """
        Update the harmony memory if necessary with the given harmony. If the given harmony is better than the worst
        harmony in memory, replace it. This function doesn't allow duplicate harmonies in memory.
    """
    if (considered_harmony, considered_fitness) not in harmony_memory:
        worst_index = None
        worst_fitness = float('+inf')
        for i, (harmony, fitness) in enumerate(harmony_memory):
            if  fitness < worst_fitness:
                worst_index = i
                worst_fitness = fitness
        if considered_fitness > worst_fitness:
            harmony_memory[worst_index] = (considered_harmony, considered_fitness)
    return harmony_memory



# fill harmony_memory using random parameter values
"""
    Initialize harmony_memory, the matrix (list of lists) containing the various harmonies (solution vectors). Note
    that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
    merely stores previous harmonies.
"""
for i in range(0, hms):

    docs_chosen=[]
    harmony = {}
    chosen_docs_numbers = random.sample(range(1, len(corpus.keys())), NUM_CLUSTERS)
    for doc_number in chosen_docs_numbers:
        docs_chosen.append ((key for key,value in docs_vs_number.items() if value== doc_number).next())

    cluster_count=0 
    for doc in corpus.keys():

        if(doc in docs_chosen):
            harmony[doc]= cluster_count
            cluster_count=cluster_count+1
        else:
            harmony[doc]=(random.randint(0,NUM_CLUSTERS-1))

    fitness, cluster_centres = get_fitness_DB(harmony,dense)
    harmony_memory.append((harmony, fitness))

# create max_imp improvisations
num_imp = 0
while(num_imp < max_imp):
    # generate new harmony
    #harmony = dict()
    docs_chosen=[]
    
    chosen_docs_numbers = random.sample(range(1, len(corpus.keys())), NUM_CLUSTERS)
    for doc_number in chosen_docs_numbers:
        docs_chosen.append ((key for key,value in docs_vs_number.items() if value== doc_number).next())
    #print docs_chosen
    cluster_count=0 
        
    for doc in corpus.keys():
        if(doc in docs_chosen):
            harmony[doc]= cluster_count
            cluster_count=cluster_count+1
        else:
            if random.random() < hmcr:
                #memory_consideration(harmony, doc)
                """
                Randomly choose a note previously played.
                """
                memory_index = random.randint(0, hms - 1)
                harmony[doc]= harmony_memory[memory_index][0][doc]
                
                if random.random() < par:
                    #print harmony
                    #raw_input("Harmony printed 272")
                    harmony[doc]=pitch_adjustment(harmony, doc, cluster_centres, dense)
            else:
                harmony[doc]=(random.randint(0,NUM_CLUSTERS-1))
    fitness, cluster_centres = get_fitness_DB(harmony,dense)

    harmony_memory=update_harmony_memory(harmony, fitness, harmony_memory)

    num_imp += 1

# return best harmony
best_harmony = None
best_fitness = float('-inf')
for harmony, fitness in harmony_memory:
    if fitness > best_fitness:
        best_harmony = harmony
        best_fitness = fitness

centers = np.array(cluster_centres)

# # K-Means based fuzzy-clustering
# cluster = KMeans(n_clusters=NUM_CLUSTERS, init=centers)
# distances = cluster.fit_transform(dense)
# distances = np.power(distances + 0.001, -1) 
# fuzzy_weights = distances
# fuzzy_weights = fuzzy_weights/np.reshape(np.sum(fuzzy_weights, axis=1), (len(fuzzy_weights), 1))
'''
Fuzzy-Clustering using fuzzy C-means algorithm

'''
def fuzzy_c_means(tfs, NUM_CLUSTERS, centers = np.zeros((0)), m = 1.01, stop_criterion = 0.01):

    if (centers.shape != (NUM_CLUSTERS, tfs.shape[1])):
        centers = tfs[[randrange(len(docs)) for i in range(NUM_CLUSTERS)]]

    # calculate fuzzy weights for each doc-cluster pair
    def weight_updates(centers):
        fuzzy_weights = np.power(euclidean_distances(tfs, centers) + 1, 2/(m-1))
        fuzzy_weights = np.power(fuzzy_weights/np.sum(fuzzy_weights, axis = 1).reshape(len(docs), 1), -1)
        return fuzzy_weights

    # calculate new cluster centers
    def centre_updates(fuzzy_weights):
        centers = np.zeros((NUM_CLUSTERS, tfs.shape[1]))
        normalization_value = np.zeros((NUM_CLUSTERS, 1))
        for j in range(len(docs)):
            fuzzy_belonging = np.power(fuzzy_weights[j, :].reshape(NUM_CLUSTERS, 1), m)
            centers = centers  + fuzzy_belonging*tfs[j]
            normalization_value = normalization_value + fuzzy_belonging

        centers = centers/normalization_value
        return centers

    # FCM algorithm
    new_fuzzy_weights = weight_updates(centers)
    while(True):
        fuzzy_weights = new_fuzzy_weights
        centers = centre_updates(fuzzy_weights)
        new_fuzzy_weights = weight_updates(centers)

        #stopping condition: maximum percentage change in fuzzy weight < stopping criterion
        if np.max(np.absolute((new_fuzzy_weights - fuzzy_weights)/fuzzy_weights)) < stop_criterion:
            break

    return [centers, fuzzy_weights]


dense = np.array(dense)
[centers, fuzzy_weights] = fuzzy_c_means(tfs = dense, NUM_CLUSTERS = NUM_CLUSTERS, centers = np.array(centers))

# Cluster evaluation
handClassification = np.zeros(tfs.shape[0])
TOPICS = ['life-style', 'business', 'entertainment', 'tech', 'sports']

for i, topic in enumerate(TOPICS):
    hardTopicClass = []
    clusterCorrelation = np.zeros((5));
    for f in os.listdir(os.path.join(mypath, topic)):
        if f in corpus.keys():
            hardTopicClass.append(corpus.keys().index(f)) 
            clusterCorrelation = clusterCorrelation +  fuzzy_weights[corpus.keys().index(f)]

    clusterValue = clusterCorrelation.argmax()
    for doc in hardTopicClass:
        handClassification[doc]=clusterValue

a=0
d=0
N=0
for i in corpus.keys():
    assigned_cluster= np.argmax(fuzzy_weights[corpus.keys().index(i)])
    for j in corpus.keys():
        N=N+1
        if i==j:
            continue
        temp= np.argmax(fuzzy_weights[corpus.keys().index(j)])
        if temp==assigned_cluster:
            if(handClassification[corpus.keys().index(i)])==(handClassification[corpus.keys().index(j)]):
                a=a+1
        else:
            if (handClassification[corpus.keys().index(i)])!=(handClassification[corpus.keys().index(j)]):
                d=d+1
OmegaIndex=float(a+d)/N
print a,d,N
print "Omega Index: ", OmegaIndex

'''
This is the Information Retrieval system. It takes a document name as a query and returns the top 10 documents that are closest to this queried document. Make sure that a valid document name from the datasets is input as query.
'''

#IR system
#takes input query and converts to doc
while(1):
    INPUT_QUERY = raw_input("Enter query")
    query=''
    for i in INPUT_QUERY.lower().translate(None, punctuation).split():
        if i not in stop:
            if wordnet_lemmatizer.lemmatize(i) in feature_names:
                query = query + ' ' + wordnet_lemmatizer.lemmatize(i)
    if(query==''):
        print "Does not match"
        continue 

    print query

    tfs_query = tfidf.transform(corpus.values()+[query])

    #to store the cosine distance of each center from the query doc 
    query_center_distances = cosine_similarity(tfs_query[tfs_query.shape[0]-1],centers)[0]
    query_center_distances = query_center_distances/np.sum(query_center_distances)
        
    documentSimilarity = cosine_similarity(query_center_distances, fuzzy_weights).flatten()
    topResults = (-documentSimilarity).argsort()[:10]

    print documentSimilarity[topResults]
    print docs[topResults]



