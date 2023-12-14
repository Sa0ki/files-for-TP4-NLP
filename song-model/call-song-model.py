from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Specify the path to the directory containing the model files
model_path = "."

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Examples of texts to test
text_examples = ["It's a beautiful day.", "I don't like this product."]

# Tokenize and encode the texts
input_texts = tokenizer(text_examples, return_tensors="pt", padding=True, truncation=True)

# Perform prediction
with torch.no_grad():
    outputs = model(**input_texts)

# Retrieve class scores
logits = outputs.logits

# Convert scores to probabilities using softmax
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Retrieve predicted class (index of the class with the highest probability)
predicted_class_indices = torch.argmax(probabilities, dim=-1).tolist()

def map_prediction_to_label(prediction_id):
    labels_mapping = {"Class 1": 'Rock', "Class 2": 'Pop', "Class 3": 'Rap', "Class 4": 'Electro', "Class 5": 'Autre'}

    return labels_mapping.get(prediction_id, 'Inconnu')  # Par défaut, retournez 'Inconnu' si l'ID n'est pas dans le mapping

# Display the results
for i, example in enumerate(text_examples):
    predicted_class_index = predicted_class_indices[i]
    predicted_class_name = f"Class {predicted_class_index}"
    print(f"Text: '{example}' | Predicted Class: {map_prediction_to_label(predicted_class_name)}")
