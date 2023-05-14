import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# Sample document
document = "I love playing football. Football is a popular sport worldwide."

# Tokenization - Split the document into sentences and words
sentences = sent_tokenize(document)
words = word_tokenize(document)

# POS Tagging - Assign part-of-speech tags to words
pos_tags = pos_tag(words)

# Stop words removal - Remove common words that do not carry much meaning
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.casefold() not in stop_words]

# Stemming - Reduce words to their base or root form
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Lemmatization - Convert words to their base or dictionary form
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

# Print the results
print('----------------------------------------------------------------------------')
print("Original Document:")
print(document)
print('----------------------------------------------------------------------------')
print("\nTokenization:")
print(words)
print('----------------------------------------------------------------------------')
print("\nPOS Tagging:")
print(pos_tags)
print('----------------------------------------------------------------------------')
print("\nStop Words Removal:")
print(filtered_words)
print('----------------------------------------------------------------------------')
print("\nStemming:")
print(stemmed_words)
print('----------------------------------------------------------------------------')
print("\nLemmatization:")
print(lemmatized_words)
print('----------------------------------------------------------------------------')

