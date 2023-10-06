# NLTK

Explanation of Concepts:

1. **Importing NLTK and Specific Modules**:

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```

Here, we import the necessary modules from NLTK for various NLP tasks, including tokenization, stopword removal, stemming, lemmatization, frequency distribution, and sentiment analysis.

2. **Downloading NLTK Resources**:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

These lines download specific NLTK resources like tokenizers, stopwords, WordNet, and the VADER lexicon for sentiment analysis.

3. **Sample Text**:

```python
text = "NLTK is a leading platform for building Python programs to work with human language data. " \
       "It provides easy-to-use interfaces to lexical resources like WordNet, along with a suite of " \
       "text processing libraries for classification, tokenization, stemming, tagging, parsing, and " \
       "semantic reasoning."
```

This is the text that we will be using for demonstration purposes.

4. **Tokenization**:

```python
tokens = word_tokenize(text)
sentences = sent_tokenize(text)
```

- `word_tokenize()` breaks the text into individual words.
- `sent_tokenize()` splits the text into sentences.

5. **Stopword Removal**:

```python
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
```

- We download a set of English stopwords and then remove them from the list of tokens.

6. **Stemming**:

```python
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
```

- We use the Porter Stemmer to perform word stemming on the filtered tokens.

7. **Lemmatization**:

```python
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
```

- We use WordNet Lemmatizer to perform lemmatization on the filtered tokens.

8. **Frequency Distribution**:

```python
freq_dist = FreqDist(tokens)
```

- `FreqDist()` computes the frequency distribution of words in the text.

9. **Sentiment Analysis**:

```python
analyzer = SentimentIntensityAnalyzer()
sentiment = analyzer.polarity_scores(text)
```

- We use the VADER sentiment analysis tool to analyze the sentiment of the text.

10. **Displaying Results**:

The code concludes with printing the results of each NLP task.

