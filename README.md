# Multilabel-Classification-of-arXiv-Paper-Summarizations


1) The type of your best performing model (e.g., Decision Tree)

Linear SVC 
HPs -> {'C': 1, 'penalty': 'l2'}


2) A short description of any data preprocessing you did (e.g., removing stopwords, cleaning up labels, etc.)

TF-IDF -> removed english stop words, 2000 dim, ngram_range=(1, 2)

experimented with PCA 2000 ->250 dims, followed by min-max scaling to [0,1] range

also experimented with classification on ALL CLASSES and by removing classes with less than 1% occurances.

3) A short description of the features you used for the best performing model 

As described above, but without PCA and min-max scaling. (described in report)

4) Your micro and macro F1 scores for your best performing model

Linear SVC with HPs tuned

Test F1 Macro: 0.405197406542662

Test F1 Micro: 0.7718218872567869


4) Anything else you’d like me to know about your code, such as any known bugs or mistakes, or just rationale for why you did something a certain way.

some code for PCA, Min-max is now commented out. since its no longer used.



