import pandas as pd
import hashlib
from collections import Counter
import numpy as np
import scipy
from sklearn import preprocessing
from collections import Counter 
from sklearn import manifold
from matplotlib import pyplot as plt
import ot


df = pd.read_csv("IRAhandle_tweets_1.csv")
filenames = ["IRAhandle_tweets_"+str(i)+".csv" for i in range(2,10)]
for fn in filenames:
    df = df.append(pd.read_csv(fn))

cont_list = list(df.content)

#in the following hashing all the tweets to into integers to create a unique index

string_list = [str(t) for t in cont_list]
sh = [int(hashlib.md5(j).hexdigest(), 16)%100000000000000 for j in string_list]
#no collisions
df['sh'] = sh

#now need a function which takes a user, and returns a list of the most common tweets. 

def get_tweets_list(handle):
    df_temp = df[df.author==handle]
    return(list(df_temp.sh))

just_tweets_df = pd.DataFrame(columns = ['author', 'tweethash'])

handles = list(set(df.author))
for hndl in handles:
    just_tweets_df.loc[len(just_tweets_df)] = [hndl, get_tweets_list(hndl)] 


def lists_to_metric(LL, top):  
    exponent = 4   #We add an a small eta and an exponent of the markov matrix to get rid help connect the similarity a bit better
    eta = 0.03
    flat_list = [item for sublist in LL for item in sublist]   
    flc = Counter(flat_list)
    common_rt = set([hh[0] for hh in flc.most_common(top)])
    threshold = flc.most_common(top)[top-1][1]
    LS = [set(l) for l in LL]
    LSS = [l.intersection(common_rt) for l in LS] 
    BL = [] 
    labels = list(common_rt)
    M = np.zeros([len(LL),len(labels)])
    for i in range(len(LL)):
        for p in LL[i]:
            if p in labels: M[i,labels.index(p)]=1
    SM = preprocessing.normalize(M, norm='l1')   #Now it's a right stochastic matrix, ie. each row sums to 1.  
    MSM = M.transpose().dot(SM) #This matrix is now the transition probability matrix.
    MSM = preprocessing.normalize(MSM, norm='l1')
    Mexp = np.linalg.matrix_power(MSM,exponent)
    NMSM = preprocessing.normalize(MSM+eta*Mexp, norm='max')
    distance_metric = -np.log(NMSM)/2
    return(NMSM,distance_metric,labels,threshold)



listlist = list(just_tweets_df.tweethash)
tweet_metric = lists_to_metric(listlist, 3000)  

from numpy import inf
dist_metric = tweet_metric[1]

dist_metric[dist_metric== inf] = 50

labels = tweet_metric[2]




def convert_hot(l, labels):                 # each user now has a binary vector indexed by retweet, with '1' if user retweeted
    cl = [k in l for k in labels]
    return(np.array([int(c) for c in cl]))

retweet_list = list(just_tweets_df.tweethash)
hotvecs = [convert_hot(i, labels) for i in retweet_list]
total_selected = [a.sum() for a in hotvecs]
just_tweets_df['hotvecs'] = hotvecs
just_tweets_df['hot_rts'] = total_selected

df8=just_tweets_df[just_tweets_df.hot_rts>4]  # produced 1346 accounts

df8 = df8.sample(800)  #starts to get heavy about 800-900 so took a sample

def get_acc_cat(handle):
    df_temp = df[df.author==handle]
    return(df_temp['account_category'].iloc[0])


cats = [get_acc_cat(hndl) for hndl in df8.author]


def get_W(v1,v2,dsq):  #vector1, vector2, and the metric to be used
    L = len(v1)
    inds=[i for i in range(L) if (v1[i]+v2[i])>0]
    lv1 = np.asfarray(v1[inds])
    lv2 = np.asfarray(v2[inds])
    lv1 = lv1/lv1.sum()
    lv2 = lv2/lv2.sum()
    ldsq = dsq[:,inds][inds]
    try : mm = ot.emd(lv1,lv2,list(ldsq))
    except : return(25)  #I believe the errors are for when vecs are empty
    value = (mm*ldsq).sum()
    return(value)

list_of_hot_vecs = df8.hotvecs

metric_on_users = [[get_W(p,q, tweet_metric[1]) for p in list_of_hot_vecs] for q in list_of_hot_vecs]


#floyd warshall may or may not do anything but run it anyways 
fw = scipy.sparse.csgraph.floyd_warshall(metric_on_users)
sfw = (fw+fw.transpose())/2
view_metric = pd.DataFrame(metric_on_users, columns = df8.author)
view_metric.index=df8.author

cats = [get_acc_cat(hndl) for hndl in df8.author]
LABEL_COLOR_MAP = {'RightTroll' : 'r',
                   'LeftTroll' : 'b',
                   'HashtagGamer' : 'c',
                   'Unknown' : 'g', 
                   'NonEnglish' : 'y',
                   'Fearmonger' : 'k',
                   'NewsFeed' : 'm',
                   }
colors = [LABEL_COLOR_MAP[q] for q in cats]



mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
pos = mds.fit(sfw).embedding_

plt.scatter(pos[:,0], pos[:,1],c = colors) 
plt.show()

mds3 = manifold.MDS(n_components=3, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
pos3 = mds3.fit(sfw).embedding_


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos3[:,0], pos3[:,1],pos3[:,2], c=colors) 
plt.show()

view_metric['x'] = pos[:,0]
view_metric['y'] = pos[:,1]
view_metric['theta'] = np.arctan(pos[:,1]/pos[:,0])
view_metric['cat'] = cats
angle_sort = view_metric.sort_values(by='theta')
angle_sort.to_csv("metric.csv")

