import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = open('test.csv', encoding='utf-8').read()
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
tokenized_word = word_tokenize(cleaned_text, "english")

final_word = []
for words in tokenized_word:
    if words not in stopwords.words("english"):
        final_word.append(words)

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
        word, emotion = clear_line.split(":")
        if word in final_word:
            emotion_list.append(emotion)
print(emotion_list)
w = Counter(emotion_list)
print(w)


def sentimental_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        print("negative sentiment")
    elif pos > neg:
        print("positive sentiment")
    else:
        print("neutral sentiment")


sentimental_analyse(cleaned_text)

fig, axl = plt.subplots()
axl.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
