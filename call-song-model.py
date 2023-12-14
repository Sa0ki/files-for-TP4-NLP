from transformers import BertForSequenceClassification, BertTokenizer

# Spécifier le chemin vers le dossier du modèle Transformers
model_path = "final_model.pth"

# Charger le modèle et le tokenizer depuis le dossier du modèle
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Mettre le modèle en mode d'évaluation
model.eval()

# Exemple de texte de test
text_example = "This is a sample sentence for testing."

# Tokeniser le texte
tokens = tokenizer(text_example, return_tensors="pt")

# Effectuer une prédiction
with torch.no_grad():
    output = model(**tokens)

# Afficher les résultats
print("Output:", output)
