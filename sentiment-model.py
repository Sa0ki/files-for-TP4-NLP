import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense

# Télécharger les données nécessaires pour l'analyse de sentiments
nltk.download('movie_reviews')  # Télécharge les critiques de films en anglais (positives et négatives)
nltk.download('punkt')  # Télécharge les modèles et données pour le tokenizer (séparateur de mots) de NLTK.
nltk.download('wordnet')  # Télécharge WordNet, une base de données lexicale de l'anglais.

# Charger le jeu de données movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Mélanger les documents
np.random.shuffle(documents)

# Prétraitement du texte
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(words):
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Diviser les données en ensembles d'entraînement et de test
train_set, test_set = train_test_split(documents, test_size=0.2, random_state=42)

X_train, y_train = zip(*train_set)
X_test, y_test = zip(*test_set)

X_train = [preprocess_text(words) for words in X_train]
X_test = [preprocess_text(words) for words in X_test]

# Tokenisation
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Remplissage des séquences pour qu'elles aient toutes la même longueur
max_length = 100
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post', truncating='post')

# Encodage des étiquettes
label_binarizer = LabelBinarizer()
y_train_encoded = label_binarizer.fit_transform(y_train)
y_test_encoded = label_binarizer.transform(y_test)

# Construction du modèle LSTM
embedding_dim = 100
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train_padded, y_train_encoded, epochs=10, batch_size=64, validation_data=(X_test_padded, y_test_encoded))

# Évaluation du modèle
y_pred = model.predict(X_test_padded)
y_pred_binary = (y_pred > 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred_binary))

# Sauvegarde du modèle en .h5
model.save('sentiment-model.h5')

# Sauvegarde du tokenizer
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

# Adapter le Tokenizer aux textes d'entraînement
tokenizer.fit_on_texts(X_train)

# Sauvegarder le Tokenizer dans un fichier
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)
