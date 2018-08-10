

I used my own method to cluster the data, based only on retweets.  
I gathered the most retweeted 3000 tweets. 
I created a similiarity metric on these, with stronger affinity between two tweets indicating that if a user had retweeted one, the user was more likely to retweet the other.   
Based on this, I created a distance metric on the tweets.  

For users, I selected the subset of users that had retweeted at least 5 of the most common tweets.  There were 1346 or these. I chose a random sample of 800.   For each user, I created a probability vector on the space of tweets space based on which tweets they had retweeted, and then computed the Wasserstein distance based on the metric determined on the tweets.  
Then, I used MDS to project the resulting metric into both 3d and 2d.  The pictures are slightly different, but the main clusters are clearly visible.  
 
 The metric data is saved in metric.csv, with these accounts listed in angular order determined by 2d MDS plot.  
 
 These are color coded according to the labels that were previously given: 'RightTroll' is red, 'LeftTroll' is blue, 'HashtagGamer' is cyan,'Unknown' is green, 'NonEnglish' is yellow ,  'Fearmonger' is black, and 'NewsFeed' is magenta. 

This is a view of the 3d image

![Image of 3d metric](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/3dfull1.png)

Then we have a 2d image
![Image of 2d metric](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/2dfull.png)

We zoom around the 2d image, starting at the top and going clockwisse
![Image of 2d metric zoom1](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/2dzoom1.png)
![Image of 2d metric zoom2](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/2dzoom2.png)
![Image of 2d metric zoom3](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/2dzoom3.png)
![Image of 2d metric zoom4](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/2dzoom4.png)
![Image of 2d metric zoom5](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/2dzoom5.png)
![Image of 2d metric zoom6](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/2dzoom6.png)
![Image of 2d metric zoom7](https://github.com/DeepStateLearning/russian-troll-tweets/blob/master/2dzoom7.png)
