# README

This small project attempts to use Classical ML applied to NLP, to then compare the results with creating a DL model from scratch, vs using transformers. We use the same IMDB movie reviews dataset for each one. 
This project is purely motivated by curiosity. I was keen to optimize each approach and compare which model to use from a business perspective, balancing computing costs and model efficiency. 

Deep Learning models and Transformers are to be found in different repositories. This one focuses on classical ML. 


## How I started and what I did wrong. 


Machine Learning models are more often than not hungry for data, so I decided to increase the size of my original dataset. 

I went to have a look on kaggle and hugging face and ended up finding a dataset full of movie reviews.  
I cleaned this new dataset, and added it's train and validation set to my training set. 
Nevertheless, I was thinking of adding more general sentiment analysis data, in order to make my data set massive. I thought my models would then be able to better comprehend the movie reviews. After a lot of research, I decided to abandon this thought. Perhaps this could be done in further research. 

After a lot of training, I finally obtained 0.96 macro f-1 score on my test set, which lead me to check if something was wrong. It seemed too good to be true. 

I changed the random state of my splits and added a function that makes every letter lowercase in my pre-processor (just in case), but still my results were strangely high. However, at this time I was evaluating on a new test set that I had created of after having combined the datasets. I didn't know that I had to evaluate only on the rotten tomatoes test set, even if I added data. 

So I then went back to the partitioning of my data at the begining, and noticed something illogical. I was adding the rotten tomatoes test set and concatenating all of the data, and THEN performing a split once I added the new data. This thus explains why my results were so high. 


## How I corrected my mistake


After having understood this, I rectified my splits. My train data now contains the train and validation set of the rotten tomatoes, and all the additional data. I left the rotten_tomatoes test set untouched. I would only use it to evaluate my model. 

My scores then went from 0.96 to 0.76 on test set. Thus, I added even more data, reaching 70k reviews in my train set. 

I also was careful to verify using cosin similarity if there were any duplicate rows. In orther words, I wanted to check if I had added instancies that were already present in the rotten tomatoes dataset. As a result, I dropped 8 instancies, that had above my 0.9 similarity threshold. 

To optimize my results, I thought of using SKOPT for hyper-Parameter optimization, but after hours of struggling to find out the problem, I understood that numpy and skopt were not compatible. So I switched to optuna. Now, instead of the model evaluating scores on different specific hyperparameters in the grid, it checks all combinations of parameters amongst a RANGE of parameters. It took longer to run, but returned better results. 



## Models: (all of them were fine tuned to see which one provided the best score)


Model 1: Random Forest - 0.76
Model 2: Naive Bayes - 0.79
Model 3: SVC - 0.76
Model 4: SGDClassifier - 0.77


Indeed, like concluded in my research before starting to code I read that Naive Bayes works suprisingly well with text, and this turned out to be true. Model 2 was the best. 

Nonetheless, adding data actually turned out to make my results worse, si I went back to the original rotten tomatoes data set and obtained a score of 0.79 with MultinomialNB, fine tuning with skopt. 

I also created a soft voting classifier with all the fine tuned models (1-4), but my score didn't improve, I still got 0.79. So I decided to show only the MultinomialNB for simplicity. 


## Conclusion


In all, after multiple efforts to find and add data, I was disappointed to see that this did not help my macro f-1 score. Perhaps I would need a tremendous amount of extra data for this to work, way more then 70k reviews. 
Unsuprisingly, the Naive Bayes model worked out the best, but not by much. For future improvements, I would perhaps recommend trying to use mixture of experts. 
