from TFIDF import TFIDF
from HMMlearning import HMMlearning


def predict(name, term):
    start_date = "06/01/2016"
    end_date = "06/28/2016"
    hmmlearning = HMMlearning(name, start_date, end_date)
    probabilities = hmmlearning.get_prob()
    tfidf = TFIDF(term, "2016-06-28")
    score = tfidf.findTFIDF();
    dates, close_v, volume = hmmlearning.get_price_all("06/28/2016", "06/29/2016")
    cpi = close_v[0]
    prediction = cpi + (probabilities[0] - probabilities[1]) * score
    percentageerror = abs(prediction - close_v[1]) / prediction * 100

    print("stock name: " + name)
    print("start_date: " + start_date)
    print("end_date: " + end_date)
    print("HMM learning results: ")
    print(probabilities)
    print("tf-idf score: ")
    print(score)
    print("cpi: ")
    print(cpi)
    print("predicted price:")
    print(round(prediction, 2))
    print("actural price:")
    print(close_v[1])
    print("percentage error:")
    print(percentageerror)

#predict("GOOG", "Google")
predict("FB", "Facebook")
predict("ORCL", "Oracle")
predict("CSCO", "Cisco")
predict("NVDA", "Nvidia")
