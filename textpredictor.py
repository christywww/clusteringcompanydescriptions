from processor import Processor
import os
import pickle
import spacy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans


class TextPredictor:
    def __init__(self, tfidf = None, selector = None, kmeans = None):
        # load all models only once to save time when doing multiple predictions
        print('Loading models...')
        self.tf_transformer = tfidf if tfidf is not None else pickle.load(open('tfidf.pkl', 'rb'))
        self.feature_selector = tfidf if tfidf is not None else pickle.load(open('selector.pkl', 'rb'))
        self.kmeans_model = tfidf if tfidf is not None else pickle.load(open('kmeans.pkl', 'rb'))

    @staticmethod
    def processed_text_generator(): # utility function for model training
        NLP = spacy.load("en_core_web_sm")  # load it only once and pass it as an argument for processor to save time!

        for filename in os.listdir('data'):
            f = os.path.join('data', filename)
            if os.path.isfile(f) and filename.endswith('.txt'):
                processor = Processor(filename, NLP=NLP)
                processor.process()

                yield processor.text

    def train_tfidf(self): # train tfiffvectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer

        tf = TfidfVectorizer()
        tf_transformer = tf.fit(self.processed_text_generator())
        pickle.dump(tf_transformer, open("tfidf.pkl", "wb"))

    def train_featureselector(self): # train feature selector (VarianceThreshold)
        from sklearn.feature_selection import VarianceThreshold

        tf_transformer = pickle.load(open('tfidf.pkl', 'rb'))
        selector = VarianceThreshold(threshold=0.0002)
        data = tf_transformer.transform(self.processed_text_generator())
        selector.fit(data)
        pickle.dump(selector, open("selector.pkl", "wb"))

    def train_kmeans(self): # train kmeans model
        from sklearn.cluster import KMeans

        tf_transformer = pickle.load(open('tfidf.pkl', 'rb'))
        selector = pickle.load(open('selector.pkl', 'rb'))

        tf_idfed = tf_transformer.transform(self.processed_text_generator())
        reduced_tf_idfed = selector.transform(tf_idfed)
        kmeans = KMeans().fit(reduced_tf_idfed)
        pickle.dump(kmeans, open("kmeans.pkl", "wb"))

    def predict_single(self, stockcode): # generate prediction for a single stock (input: string)
        processor = Processor(stockcode + '.txt')
        processor.process()

        print('Predicting...')
        tf_idfed = self.tf_transformer.transform([processor.text])
        reduced_tf_idfed = self.feature_selector.transform(tf_idfed)
        result = self.kmeans_model.predict(reduced_tf_idfed)
        print('Stock', stockcode, 'belongs to group', result)


    def predict_multiple(self, stocknames): # generate predictions for multiple stocks (list of strings) at the same time + generates summary report
        NLP = spacy.load("en_core_web_sm")

        cats = defaultdict(list) # defaultdict to store categories

        print('Predicting...')
        for stock in stocknames:
            processor = Processor(stock + '.txt', NLP=NLP)
            processor.process()
            tf_idfed = self.tf_transformer.transform([processor.text])
            reduced_tf_idfed = self.feature_selector.transform(tf_idfed)
            result = self.kmeans_model.predict(reduced_tf_idfed)

            print('Stock', stock, 'belongs to group', result)
            cats[result[0]].append(stock)

        # show summary report
        print('\n=========== REPORT =========')
        for key in sorted(cats.keys()):
            print('Group ' + str(key) + ': ' + ", ".join(cats[key]))

    def predict_all(self, verbose = False): # generate predictions + summary report for all stocks
        NLP = spacy.load("en_core_web_sm")
        cats = defaultdict(list)

        print('Predicting...')
        for filename in os.listdir('data'):
            if filename.endswith('.txt'):
                stock = filename[:-4]
                processor = Processor(filename, NLP=NLP)
                processor.process()
                tf_idfed = self.tf_transformer.transform([processor.text])
                reduced_tf_idfed = self.feature_selector.transform(tf_idfed)
                result = self.kmeans_model.predict(reduced_tf_idfed)

                if verbose: print('Stock', stock, 'belongs to group', result)
                cats[result[0]].append(stock)

        # show summary report
        print('\n=========== REPORT =========')
        for key in sorted(cats.keys()):
            print('Group ' + str(key) + ': ' + ", ".join(cats[key]))


if __name__ == '__main__':
    tp = TextPredictor()
    tp.predict_all(verbose=True)