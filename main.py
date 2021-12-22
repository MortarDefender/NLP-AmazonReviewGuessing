import json
import string
import numpy as np
import pandas as pd
import seaborn as sns
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as Downscale


class ReviewClassifier():
    class __ReviewClass(Enum):
        OneStar   = 1
        TwoStar   = 2
        ThreeStar = 3
        FourStar  = 4
        FiveStar  = 5
    
    def __init__(self, trainFileName):
        self.corpus = list()
        self.classifier = None
        self.maxFeatures = 1000
        self.minWordsAmount = 1
        self.maxWordsAmount = 5
        self.maxFrequentCut = 0
        self.wordsFrequency = None
        self.downsizer = Downscale()
        self.trainFileName = trainFileName
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(self.minWordsAmount, self.maxWordsAmount), max_features=self.maxFeatures)
    
    @staticmethod
    def __disassembleReview(review):
        """ extract review text and overall """
        review = json.loads(review)
        
        text = review.get("reviewText") if review.get("reviewText") is not None and bool(review.get("verified")) else None
        summery = review.get("summary") if review.get("summary") is not None and bool(review.get("verified")) else None
        score = int(review["overall"]) if review.get("overall") is not None else 0
        
        return text, summery, score
    
    def __disassembleAllReviews(self):
        """ join all i star reviews data into a single string array """
        rateArr = [""] * len(self.__ReviewClass)
         
        with open(self.trainFileName, 'r') as trainFile:
            lines = trainFile.readlines() 
        
        for review in lines:
            text, summery, score = self.__disassembleReview(review)
            
            if score != 0 and text is not None:
                rateArr[score - 1] += text + " "
                
                if summery is not None:
                    rateArr[score - 1] += summery + " "
        
        return rateArr
    
    def __createTestArray(self):
        arr = []
        
        with open(self.trainFileName, 'r') as trainFile:
            lines = trainFile.readlines() 
        
        for review in lines:
            text, summery, score = self.__disassembleReview(review)
            
            if score != 0 and text is not None:
                if summery is not None:
                    text = text + " " + summery
                    
                text = self.__removePunctuation(text)
                text = self.__removStopWords(text)
                arr.append(text)
        
        return arr
    
    def __getAllLabels(self):
        labels = []
        
        with open(self.trainFileName, 'r') as trainFile:
            lines = trainFile.readlines() 
        
        for review in lines:
            text, summery, score = self.__disassembleReview(review)
            
            if score != 0 and text is not None:
                labels.append(score)

        return labels
    
    @staticmethod
    def __removePunctuation(text):
        """ remove the punctuation from the text given and return it """
        punctuation = str.maketrans('', '', string.punctuation)
        
        return ''.join(list(filter(lambda x: x != '', [text[row].translate(punctuation) for row in range(len(text))])))
    
    @staticmethod
    def __removStopWords(text):
        """ remove stop words from the text given """
        stopwords = ['i', 'I', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                     "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                     "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
                     'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                     'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                     'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                     'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                     'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                     'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                     'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                     "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                     "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                     'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                     'won', "won't", 'wouldn', "wouldn't"]
        
        return " ".join(list(filter(lambda x: x not in stopwords and not x.isnumeric(), text.split())))
    
    def __createMostFrequentWords(self, corpus):
        """ creates the most frequent words dictionary """
        
        if self.wordsFrequency is None:
            allWords = " ".join(corpus)
            self.wordsFrequency = defaultdict(int)
            
            for word in allWords.split():
                self.wordsFrequency[word] += 1
    
    def __removeMostFrequentWords(self, corpus):
        """ remove the max frequent words from the corpus """
        frequentWords = [item[0] for item in sorted(self.wordsFrequency.items(), key=lambda kv: kv[1], reverse=True)[:self.maxFrequentCut]]

        for i, line in enumerate(corpus):
            corpus[i] = " ".join(list(filter(lambda x: x not in frequentWords, corpus[i].split())))
        
        return corpus
    
    def __removeRedundancy(self, corpus):
        """ remove punctuation, stopwords and most common words from the corpus """
        
        for i, line in enumerate(corpus):
            corpus[i] = self.__removePunctuation(line)
            corpus[i] = self.__removStopWords(corpus[i])
        
        self.__createMostFrequentWords(corpus)
        return self.__removeMostFrequentWords(corpus)
    
    def __createBagOfWords(self):
        """ create the distinct words and words frequency dictionary """
        self.corpus = self.__createTestArray() # self.__removeRedundancy(self.__disassembleAllReviews())
        arrayOfAppearances = self.vectorizer.fit_transform(self.corpus)
        return self.downsizer.fit_transform(arrayOfAppearances)
    
    @staticmethod
    def __getPredictedData(predicted, trueResults):
        """ return a dictionary in the format of 'class_i.0_F1' and accuracy
            each item get a precentage from the overall """
        overall = len(predicted)
        test_results = {'class_1.0_F1': 0.0, 'class_2.0_F1': 0.0, 'class_3.0_F1': 0.0,
                        'class_4.0_F1': 0.0, 'class_5.0_F1': 0.0, 'accuracy': 0.0}
        
        results = [0] * 5
        
        for reviewPrediction, realScore in zip(predicted, trueResults):
            results[realScore - 1] += 1
            test_results["class_{}.0_F1".format(reviewPrediction)] += ((1 / overall) * 100)

        test_results['accuracy'] =  np.mean(predicted == trueResults) * 100
        
        return test_results       

    def __showConfusionMatrix(self, trueRes, pred):
        """ show the confusion matrix of the test given """
        confusionMatrix = confusion_matrix(trueRes, pred, labels=[x.value for x in self.__ReviewClass])
        confusionDataFrame = pd.DataFrame(confusionMatrix, index = [str(x.value) for x in self.__ReviewClass], columns = [str(x.value) for x in self.__ReviewClass])
        print(confusionDataFrame)
        sns.heatmap(confusionDataFrame, fmt="", annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()
    
    def __getBestFeatures(self, bagOfWords, maxFeaturesAmount=15):
        """ return a list of the maxFeaturesAmount of the classifer given """
        bestFeatures = list()
        kBest = SelectKBest(chi2, k=maxFeaturesAmount).fit_transform(bagOfWords, [x.value for x in self.__ReviewClass])
        
        for item in list(kBest.todok().keys())[:15]:
            bestFeatures.append(self.corpus[item[0]].split(" ")[item[1]])
        
        return bestFeatures
    
    def __fitModel(self, testFileName):
        """ fit the model per the clf given and predict according to the test file given """
        bagOfWords = self.__createBagOfWords()
        self.classifier = self.classifier.fit(bagOfWords, self.__getAllLabels()) # [x.value for x in self.__ReviewClass]
        # kBestFeatures = self.__getBestFeatures(bagOfWords)
        
        testReviewsData = list()
        testReviewsScore = list()
        
        for review in open(testFileName, 'r').readlines():
            data, summery, score = self.__disassembleReview(review)
            
            if score != 0 and data is not None:
                if summery is not None:
                    testReviewsData.append(data + " " + summery)
                else:
                    testReviewsData.append(data)
                testReviewsScore.append(score)
        
        testReviewsData = self.__removeRedundancy(testReviewsData)
        
        new_data = self.vectorizer.transform(testReviewsData)
        tfidf = self.downsizer.transform(new_data)
        predicted = self.classifier.predict(tfidf)
        
        return predicted, testReviewsScore, None # kBestFeatures
    
    def __getFitResults(self, testFileName, showMatrix=True):
        """ get the result from the fit method """
        predicted, trueResults, kBestFeatures = self.__fitModel(testFileName)
        
        if showMatrix:
            self.__showConfusionMatrix(trueResults, predicted)
        
        return self.__getPredictedData(predicted, trueResults)
    
    def __testPossibilities(self, testFileName, maxCut=1000):
        """ get the maximum possible accuracy of the model using an iteration over max frequent cut"""
        maxNCut = 0
        max_test_results = None
        
        for nCut in range(0, maxCut, 100):
            self.maxFrequentCut = nCut
                    
            current_test_results = self.__getFitResults(testFileName, False)
            
            if max_test_results is None or max_test_results['accuracy'] < current_test_results['accuracy']:
                max_test_results = current_test_results
                maxNCut = nCut
        
        self.maxFrequentCut = maxNCut
        self.__getFitResults(testFileName)
        return max_test_results
    
    def fitNaiveBayes(self, testFileName):
        """ get the result of the model over the Naive Classifier """
        self.classifier = MultinomialNB()
        return self.__testPossibilities(testFileName)
    
    def fitLogisticRegression(self, testFileName):
        """ get the result of the model over the Logistic Regression Classifier """
        self.classifier = LogisticRegression(random_state = 0)
        return self.__getFitResults(testFileName)
        # return self.__testPossibilities(testFileName)
    

def classify(train_file, test_file):
    print('starting feature extraction and classification, train data:', train_file, 'and test:', test_file)

    # return ReviewClassifier(train_file).fitNaiveBayes(test_file)
    return ReviewClassifier(train_file).fitLogisticRegression(test_file)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)
