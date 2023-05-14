from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Example document
document = "This is an example document for calculating TF and TF-IDF."

# Create the CountVectorizer for TF
tf_vectorizer = CountVectorizer()

# Fit and transform the document for TF
tf_matrix = tf_vectorizer.fit_transform([document])

# Get the feature names (terms)
feature_names_tf = tf_vectorizer.get_feature_names()

# Get the TF values
tf_values = tf_matrix.toarray()[0]

# Print the feature names and their corresponding TF values
print('-------------------------------------------------------------')
print("TF Representation:")
for term, tf in zip(feature_names_tf, tf_values):
    print(term, ":", tf)

print('-------------------------------------------------------------')


# Create the TfidfVectorizer for TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the document for TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform([document])

# Get the feature names (terms)
feature_names_tfidf = tfidf_vectorizer.get_feature_names()

# Get the TF-IDF values
tfidf_values = tfidf_matrix.toarray()[0]

# Print the feature names and their corresponding TF-IDF values
print('-------------------------------------------------------------')
print("\nTF-IDF Representation:")
for term, tfidf in zip(feature_names_tfidf, tfidf_values):
    print(term, ":", tfidf)
print('-------------------------------------------------------------')
