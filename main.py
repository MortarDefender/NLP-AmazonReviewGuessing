import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as Downscale

from sklearn.linear_model import SGDClassifier
from enum import Enum

class ReviewClassifier():
    class __ReviewClass(Enum):
        OneStar   = 1
        TwoStar   = 2
        ThreeStar = 3
        FourStar  = 4
        FiveStar  = 5
    
    
    def __init__(self, trainFileName):
        self.debug = False
        self.classifier = None
        self.minWordsAmount = 1  # 3  # 1
        self.maxWordsAmount = 5  # 10 # 2
        self.maxFrequentCut = 11 # 14
        self.arrayOfAppearances = list()        # can be temporary
        self.trainFileName = trainFileName
        self.distinctWords = dict()             # needed ??
        self.wordsFrequency = None              # needed ?? Well Yes But No
        self.downsizer = Downscale()
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(self.minWordsAmount, self.maxWordsAmount))
    
    @staticmethod
    def __disassembleReview(review):
        """ extract review text and overall """
        review = json.loads(review)
        
        text = review.get("reviewText") if review.get("reviewText") is not None else None
        score = int(review["overall"]) if review.get("overall") is not None else 0
        
        return text, score
    
    def __disassembleAllReviews(self):
        """ join all i star reviews data into a single string array """
        rateArr = [""] * len(self.__ReviewClass)
        
        with open(self.trainFileName, 'r') as trainFile:
            lines = trainFile.readlines()
            
        for review in lines:
            text, score = self.__disassembleReview(review)
            
            if score != 0 and text is not None:
                rateArr[score - 1] += text + " "
            
        return rateArr
    
    def __removeRedundancy(self, corpus):
        import string
        
        punctuation = str.maketrans('', '', string.punctuation)
        stopwords = ['i', 'I','me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
                         "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
                         'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                         'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
                         'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
                         'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                         'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 
                         'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
                         'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
                         'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                         'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
                         "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
                         "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
                         "won't", 'wouldn', "wouldn't"]
        
        for i, line in enumerate(corpus):
            ## remove punctuation
            corpus[i] = ''.join(list(filter(lambda x: x != '', [line[row].translate(punctuation) for row in range(len(line))])))
            
            ## remove stopwords
            corpus[i] = " ".join(list(filter(lambda x: x not in stopwords and not x.isnumeric(), corpus[i].split())))
            
        
        if self.wordsFrequency is None:
            self.wordsFrequency = defaultdict(int)
            ## create wordFrequency dictionary  -> can be done once
            allWords = " ".join(corpus)
            self.wordsFrequency = defaultdict(int)
            for word in allWords.split():
                self.wordsFrequency[word] += 1
        
        ## get the 1000 most common words
        frequentWords = [keyValue[0] for keyValue in sorted(self.wordsFrequency.items(), key=lambda kv: kv[1], reverse=True)[:self.maxFrequentCut]]
        
        
        ## remove from each star review this words from it
        for i, line in enumerate(corpus):
            corpus[i] = " ".join(list(filter(lambda x: x not in frequentWords, corpus[i].split())))
        
        return corpus
    
    def __removePunctuation(text):
        punctuation = str.maketrans('', '', string.punctuation)
        return ''.join(list(filter(lambda x: x != '', [text[row].translate(punctuation) for row in range(len(text))])))
        
    
    def __createBagOfWords(self):
        """ create the distinct words and words frequency dictionary """
        corpus = self.__removeRedundancy(self.__disassembleAllReviews())
        self.arrayOfAppearances = self.vectorizer.fit_transform(corpus)

        # for i, word in enumerate(self.vectorizer.get_feature_names_out()):
        #     self.distinctWords[word] = i

        # for starReview in corpus:
        #     for word in starReview:
        #         self.wordsFrequency[word] += 1

        # self.wordsFrequency = sorted(self.wordsFrequency.items(), key=lambda kv: kv[1])[:1000]
        return self.downsizer.fit_transform(self.arrayOfAppearances)
    
    @staticmethod
    def __printPrediction(testReviewsData, predicted):
        for review, score in zip(predicted, predicted):
            print('%r => %s' % (predicted, score))
    
    @staticmethod
    def __getPredictedData(predicted, trueResults):
        overall = len(predicted)
        test_results = {'class_1.0_F1': 0.0, 'class_2.0_F1': 0.0, 'class_3.0_F1': 0.0,
                        'class_4.0_F1': 0.0, 'class_5.0_F1': 0.0, 'accuracy': 0.0}
        
        results = [0] * 5
        
        for reviewPrediction, realScore in zip(predicted, trueResults):
            results[realScore - 1] += 1
            test_results["class_{}.0_F1".format(reviewPrediction)] += ((1 / overall) * 100)
        
        # for i in range(1, 6, 1):
        #     test_results["class_{}.0_F1".format(i)] = (test_results["class_{}.0_F1".format(i)] / results[i - 1]) * 100
        
        test_results['accuracy'] =  np.mean(predicted == trueResults) * 100
        
        return test_results       

    def __showConfusionMatrix(self, trueRes, pred):
        import warnings
        warnings.filterwarnings("ignore")
        
        confusionMatrix = [[]] * len(self.__ReviewClass)
        
        for index in range(len(self.__ReviewClass)):
            confusionMatrix[index] = [0] * len(self.__ReviewClass)
        
        for trueScore, predictedScore in zip(trueRes, pred):
            confusionMatrix[trueScore - 1][predictedScore - 1] += 1
        
        # deprected
        for i, line in enumerate(confusionMatrix):
            print(i, line)
        
        labels =   ["1", "2", "3", "4", "5"]

        plotFigure = plt.figure()
        axesPlot = plotFigure.add_subplot(111)
        plotFigure.colorbar(axesPlot.matshow(confusionMatrix, cmap='binary'))
        axesPlot.set_xticklabels([""] + labels)
        axesPlot.set_yticklabels([""] + labels)
        plt.show()
    
    def __fitModel(self, testFileName):
        #TODO: kBest = SelectKBest(chi2, k=15).fit_transform(self.classifier.fit(self.__createBagOfWords(), self.__ReviewClass))
        self.classifier = self.classifier.fit(self.__createBagOfWords(), [x.value for x in self.__ReviewClass])
        testReviewsData = list()
        testReviewsScore = list()
        for review in open(testFileName, 'r').readlines():
            data, score = self.__disassembleReview(review)
            
            if score != 0 and data is not None:
                testReviewsData.append(data)
                testReviewsScore.append(score)
        
        testReviewsData = self.__removeRedundancy(testReviewsData)
        
        new_data = self.vectorizer.transform(testReviewsData)
        tfidf = self.downsizer.transform(new_data)
            
        predicted = self.classifier.predict(tfidf)
        
        if self.debug:
            self.__printPrediction(testReviewsData, predicted)
        
        return predicted, testReviewsScore # , kBest
    
    def __getFitResults(self, testFileName, showMatrix=True):
        predicted, trueResults = self.__fitModel(testFileName)
        
        if showMatrix:
            self.__showConfusionMatrix(trueResults, predicted)
        
        return self.__getPredictedData(predicted, trueResults)
    
    def __testPossibilities(self, testFileName, minRange, maxRange, maxCut):
        max_test_results = None
        current_test_results = {}
        n, s, e = 0, 0, 0
        
        for nCut in range(maxCut):
            for startRange in range(1, minRange):
                for endRange in range(minRange, maxRange):
                    
                    if startRange > endRange:
                        continue
                    
                    self.minWordsAmount = startRange
                    self.maxWordsAmount = endRange
                    self.maxFrequentCut = nCut
                    
                    current_test_results = self.__getFitResults(testFileName, False)
                    
                    if max_test_results is None or max_test_results['accuracy'] < current_test_results['accuracy']:
                        print("accuracy: {} || [{}: {}] => {}".format(current_test_results['accuracy'], startRange, endRange, nCut))
                        max_test_results = current_test_results
                        s, e, n = startRange, endRange, nCut
        
        print("accuracy: {} || [{}: {}] => {}".format(max_test_results['accuracy'], s, e, n))
        return max_test_results
    
    def fitNaiveBayes(self, testFileName):
        self.classifier = MultinomialNB()
        return self.__testPossibilities(testFileName, 3, 7, 15)
        # return self.__getFitResults(testFileName)
    
    def fitLogisticRegression(self, testFileName):
        self.classifier = LogisticRegression(random_state = 0)
        return self.__getFitResults(testFileName)
    
    def fitSVM(self, testFileName):
        self.classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
        return self.__getFitResults(testFileName)
    

def classify(train_file, test_file):
    print('starting feature extraction and classification, train data:', train_file, 'and test:', test_file)

    return ReviewClassifier(train_file).fitNaiveBayes(test_file)
    # return ReviewClassifier(train_file).fitLogisticRegression(test_file)
    # return ReviewClassifier(train_file).fitSVM(test_file)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)
