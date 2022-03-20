import re
import spacy

class Processor:
    def __init__(self, filename, NLP = None):
        self.filename = filename

        if NLP == None:
            # python - m spacy download en_core_web_lg <- run this in terminal first :p
            self.NLP = spacy.load("en_core_web_sm")
        else:
            self.NLP = NLP

    def remove_icky_bits(self):
        # removing icky bits
        self.text = self.text.replace('Inc.', '')

        text = self.text.split('\n')  # split text into lines

        # process by paragraphs / sections (split by new line)

        for idx, line in enumerate(text):
            # remove undesired lines / sections (section titles / tables / regulatory section)
            splittedline = line.split()  # split line into words
            if len(splittedline) <= 5 or (len(splittedline) <= 10 and splittedline[0] == '-'): # delete section titles / tables (lines with <=5 words)
                text[idx] = ''
            if any(keyword in line for keyword in ['Schedule', 'Rule', 'Form', 'Section', 'www.']): # remove the regulatory / additional reporting info section
                text[idx] = ''

            # remove COVID-19 section too because that's not related to the company type
            if any(keyword in line.lower() for keyword in ['covid', 'covid-19', 'coronavirus']): # remove the regulatory / additional reporting info section
                text[idx] = ''

        self.text = ' '.join(text)

        # process by sentences (split by full stops)
        text = self.text.split('. ')
        self.blurb = text[0].strip() # extract blurb (first sentence)

        for idx, line in enumerate(text):
            # remove undesired sentences
            if any(keyword in line for keyword in ['shareholder', 'shareholders', 'stockholder', 'stockholders', 'stock']): # remove the regulatory / additional reporting info section
                text[idx] = ''

            if '$' in line or '%' in line: # removes sentences talking about revenue
                text[idx] = ''

        self.text = '. '.join(text)

        # process by word (split with space)
        text = self.text.split()

        for idx, word in enumerate(text):
            # remove bad encodings stuck to words
            word = word.replace('&quot;', '')
            word = word.replace('&amp;', '')
            re.sub('&#x?[0-9]*[a-z]?;', '', word) # (ie. &#x201d; &#x2019;)
            text[idx] = word

            # remove all acryonyms (full uppercase words)
            if word.isupper():
                text[idx] = ''

        self.text = ' '.join(text)

    def clean_data(self):

        sentencesintext = self.text.split('. ')
        NLP = self.NLP
        for idx, sentence in enumerate(sentencesintext):
            # do NER and merge multi-word entities (ie. New York -> NewYork)
            NERtext = NLP(sentence)
            for entity in NERtext.ents:
                if entity.label_ in ['ORG', 'GPE', 'PERSON'] and len(entity) > 1:
                    sentence = sentence.replace(str(entity), ("".join(str(entity).split())))

            # lemmaize sentence (also automatically changes it to lowercase)
            lemmatext = NLP(sentence) # have to do it again after NER grouping modifies the sentence
            sentence = [token.lemma_ for token in lemmatext if token.pos_ not in ['PUNCT', 'NUM']] # removes numbers and

            # remove stopwords + convert to lowercase
            sentence = [word.lower() for word in sentence if word not in self.NLP.Defaults.stop_words]
            sentencesintext[idx] = sentence

        self.sentencesintext = sentencesintext
        self.text = " ".join([" ".join(sentence) for sentence in self.sentencesintext])


    def process(self):
        self.f = open('data/' + self.filename, "r")
        self.text = self.f.read()
        self.f.close()

        self.remove_icky_bits()
        self.clean_data()


if __name__ == '__main__':
    processor = Processor('NLAB.txt')
    processor.process()

    print('finished')