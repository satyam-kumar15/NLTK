import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Sample text for demonstration
text = "NLTK is a leading platform for building Python programs to work with human language data. " \
       "It provides easy-to-use interfaces to lexical resources like WordNet, along with a suite of " \
       "text processing libraries for classification, tokenization, stemming, tagging, parsing, and " \
       "semantic reasoning."

# Tokenization
tokens = word_tokenize(text)
sentences = sent_tokenize(text)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Stemming
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Frequency Distribution
freq_dist = FreqDist(tokens)

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
sentiment = analyzer.polarity_scores(text)

# Display Results
print("Original Text:")
print(text)
print("\nTokenization:")
print(tokens)
print("\nSentence Tokenization:")
print(sentences)
print("\nStopword Removal:")
print(filtered_tokens)
print("\nStemming:")
print(stemmed_tokens)
print("\nLemmatization:")
print(lemmatized_tokens)
print("\nFrequency Distribution:")
print(freq_dist.most_common(5))
print("\nSentiment Analysis:")
print(sentiment)
