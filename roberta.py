from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import csv
import urllib.request
from enum import Enum

class Sentiment(Enum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2

class roBERTa():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

        #move to cuda if available
        self.model.to('cuda')

    def classifySentiment(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        encoded_input.to('cuda')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        sentiment_idx = np.argmax(scores)

        if sentiment_idx == 0:
            return Sentiment.NEGATIVE
        elif sentiment_idx == 1:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE

if __name__ == '__main__':
    llm = roBERTa()
    comment = '''
    The flight was really confortable.  The flight attendant couldn't have been nicer.  I always fly with you guys :)
    '''
    print(llm.classifySentiment(comment))





