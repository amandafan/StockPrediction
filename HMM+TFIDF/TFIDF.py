from readnews import ReadNews
import math


class TFIDF:

    '''
    This class takes a string term and a string formated time,
    then find all the related news within the time range.
    Finally it calculate the TF-IDF score for that term
    Sample usage:
    tfidf = TFIDF("Google", "2016-06")
    score = tfidf.findTFIDF();
    print(score)
    '''
    def __init__(self, term, time):
        file = ReadNews();
        self.newsList = file.get_news(time);
        self.t = term
        # print(self.newsList)
        # print(self.newsList.__len__())

    # TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in the document).
    # Find the sum of all TF for d in newsList
    def findTFIDF (self):
        tfscore = 0.0
        count = 0
        if self.newsList.__len__ == 0:
            return 0
        else:
            for d in self.newsList:
                if self.t in d:
                    list_of_terms = d.split(' ')
                    tfscore += d.count(self.t) / list_of_terms.__len__()
                    count += 1
                    # print(tfscore)
                    # print(count)
            if count == 0:
                return 0
            idfscore = math.log(self.newsList.__len__() / count)
            # print(idfscore)
            return tfscore * idfscore

    # IDF(t, D) = log_e(Total number of documents in D / Number of documents with term t in it). [9]
    def findIDF (self):

        return 0

