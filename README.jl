READ ME

To run the code and test different feature sets use Main.ipynb 

From there you can run all cells and this will run with my best feature set

If you would like to use other features like cut, ngrams and nounphrases you just need to uncomment the sections of code realting to these.

To include stemming:
 Uncomment these lines in the second cell in the appropriate loops:
    # X_train = remove_stopwords_and_stem(X_train) 
    # X_eval = remove_stopwords_and_stem(X_eval)
    # X_test[x] = remove_stopwords_and_stem(X_test[x])
To include cut:
 Uncomment these lines in the second cell:
    # X_train = cut(X_train, 0.002) #uncomment for frequency cut
    # X_eval = cut(X_eval, 0.002) #uncomment for frequency cut
    # X_test = cut(X_test, 0.002) #uncomment for frequency cut

To include noun phrases:
 Uncomment these lines in the second cell:
    # nounPhrasestrain = run(X_train) #uncomment this for noun phrases
    # nounPhraseseval = run(X_eval) #uncomment this for noun phrases
    # nounPhrasestest = run(X_test) #uncomment this for noun phrases

 You will also need to uncomment the appropriate lines in the feature set creation cell to include these features in the feature set.
 They are followed by print("done5"), print("done6"), print("done7") comments.
 
 

To include ngrams:
 Uncomment the appropriate lines in the feature set creation cell which are followed by print("done3"), print("done4") and print("done5)") comments.

You will also need to uncomment the lines for normalization and formatting in the 5th cell.


