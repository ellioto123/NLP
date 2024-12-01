import nltk

def cut(reviewset,cutpercent):
    # Gather all the words in the review set
    allWords = []
    for review in reviewset:
        for word in review:
            allWords.append(word)

    fdist1 = nltk.FreqDist(allWords) #get a frequency distribution of all the words
    numTokens = len(fdist1) #get the number of unique words/tokens
    numremove = int(numTokens * cutpercent) #remove 5% of the tokens
    tempremove = fdist1.most_common(numremove) #get the most common words in range of numremove
    tokens = [token for token, _ in tempremove] #extract only the token ignoring the count 
    tempremove = fdist1.most_common()[:-numremove-1:-1] #get the least common words in range of numremove
    tokens += [token for token, _ in tempremove] #extract only the token ignoring the count again
    tokens = set(tokens) #convert this to a set so it is only unique values

    temp = []
    for review in reviewset:
        tempReview = [word for word in review if word not in tokens]
        temp.append(tempReview)

    return temp