#!/usr/bin/env python
# coding: utf-8

# In[24]:


from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.preprocessing.text import Tokenizer


# In[25]:


loaded_model = load_model('sentiment-model.h5')


# In[26]:


new_texts = [
    "I absolutely love this product, it was good.",
    "The service was fantastic, and the staff were very helpful.",
    "This movie is amazing, the plot is captivating and the acting is superb.",
    "The book was a complete letdown, I couldn't get into the story at all.",
    "The food at the restaurant was terrible, I wouldn't recommend it to anyone."
]


# In[27]:


with open('tokenizer.pkl', 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

# Prétraitement des nouvelles données
max_words = 100
new_sequences = loaded_tokenizer.texts_to_sequences(new_texts)
padded_sequences = pad_sequences(new_sequences, maxlen=max_words)


# In[28]:


# Prédiction avec le modèle chargé
predictions = loaded_model.predict(padded_sequences)

# Affichage des prédictions
for i, text in enumerate(new_texts):
    sentiment = "positif" if predictions[i] >= 0.5 else "négatif"
    print(f"Texte: '{text}' | Prédiction de sentiment: {sentiment} (Confiance: {predictions[i][0]:.4f})")


# In[ ]:




