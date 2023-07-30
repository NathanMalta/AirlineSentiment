import openai
from enum import Enum

class Sentiment(Enum):
    POSITIVE = 'Positive'
    NEGATIVE = 'Negative'
    UNRELATED = 'Unrelated'
    INVALID = 'Invalid'

class LLM():
    '''A class to handle classifying comments' sentiment via few shot usage of the 
       ChatGPT (gpt-3.5-turbo) model from OpenAI via their API.
    '''

    def __init__(self, key_path: str):
        #load openai string from file
        openai.api_key = [line.rstrip('\n') for line in open(key_path)][0]

        self.systemPrompt = '''
        You job is to classify user comments from the facebook page of {}.
        Classify comments into the following sentiments based on their feelings toward the airline or it's services:
        Positive, Negative, Unrelated.
        Respond only with the label.
        '''

    def _getAIResponse(self, next_input: str, airline_name: str):
        '''Use the OpenAI API to get a response from ChatGPT
        '''

        systemPrompt = self.systemPrompt.format(airline_name)

        conversation = [
                #The system prompt telling ChatGPT it is a sentiment analysis task
                {"role": "system", "content": systemPrompt},

                #some few shot examples of the task
                {"role": "user", "content": "My flight was delayed!"},
                {"role": "assistant", "content": "Negative"},
                {"role": "user", "content": "Check out my website"},
                {"role": "assistant", "content": "Unrelated"},
                {"role": "user", "content": f"{airline_name} rocks"},
                {"role": "assistant", "content": "Positive"},

                #the actual comment to be classified
                {"role": "user", "content": next_input}
        ]

        out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
            max_tokens=100,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
        )
        response = out['choices'][0]['message']['content']
        return response

    def classifySentiment(self, comment: str, airline_name: str):
        '''Use ChatGPT to run sentiment analysis on a certain comment
        '''
        response = self._getAIResponse(comment, airline_name)

        for sentiment in [Sentiment.NEGATIVE, Sentiment.POSITIVE, Sentiment.UNRELATED]:
            if sentiment.value in response:
                return sentiment

        print(f'invalid response: {response}')
        return Sentiment.INVALID

if __name__ == '__main__':
    llm = LLM('key-openai.txt')

    #a few test cases to make sure the sentiment analysis is working properly
    assert llm.classifySentiment('I am never flying with you again!', 'Southwest Airlines') == Sentiment.NEGATIVE
    assert llm.classifySentiment('Do you guys serve pizza?' ,'Delta Airlines') == Sentiment.UNRELATED
    assert llm.classifySentiment('All my flights have been perfect.  Nice work team!', 'Spirit Airlines') == Sentiment.POSITIVE

    print('All tests passed - Sentiment Analysis seems to be working properly.')


