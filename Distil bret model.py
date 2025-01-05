#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


# Load and prepare data
data = pd.read_csv('/Users/pillisachethan/Desktop/NLP project/test_data.csv')
data['post'] = data['post'].fillna('')

# Remove non-alphabetical characters and lowercase the text
data['post'] = data['post'].apply(lambda x: re.sub(r'[^a-z\s]', '', x.lower()) if isinstance(x, str) else x)

# Initialize stopwords set and PorterStemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Function to remove stopwords and apply stemming
def preprocess_text(text):
    # Remove stopwords
    filtered_words = [word for word in text.split() if word not in stop_words]
    # Apply stemming
    stemmed_text = ' '.join(ps.stem(word) for word in filtered_words)
    return stemmed_text

# Apply the preprocessing to the 'post' column
data['post'] = data['post'].apply(preprocess_text)
data['text'] = data['post'].str.lower()


# Encode labels
label_encoder = LabelEncoder()
data['labels'] = label_encoder.fit_transform(data['subreddit'])

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['labels']
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len)
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label)
        }

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))

# Create Dataset for training and testing sets
train_dataset = CustomDataset(train_data, tokenizer, max_len=5)
test_dataset = CustomDataset(test_data, tokenizer, max_len=5)

# Training arguments with optimizations
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Start with 1 epoch
    per_device_train_batch_size=100,  
    per_device_eval_batch_size=100,
    gradient_accumulation_steps=2,  
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=1,
    fp16=False,  # Disable fp16 if there's no support for it
    bf16=True,  # Enable bfloat16 for faster training if supported
)


# Initialize Trainer with evaluation and early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save model, tokenizer, and label encoder
model.save_pretrained('./fine_tuned_distilbert')
tokenizer.save_pretrained('./fine_tuned_distilbert')
joblib.dump(label_encoder, './fine_tuned_distilbert/label_encoder.joblib')



# In[2]:


# Evaluate the model on the test set
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Reload for prediction
tokenizer = DistilBertTokenizer.from_pretrained('./fine_tuned_distilbert')
model = DistilBertForSequenceClassification.from_pretrained('./fine_tuned_distilbert')
label_encoder = joblib.load('./fine_tuned_distilbert/label_encoder.joblib')

# Function for individual text prediction
def predict_behavior(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=5)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    subreddit = label_encoder.inverse_transform([predicted_label])[0]
    return subreddit

# Function to predict on test set and add predictions
def predict_on_test_set(test_data):
    model.eval() 
    predictions = []
    
    for i in range(len(test_data)):
        text = test_data.iloc[i]['post']
        predicted_subreddit = predict_behavior(text)
        predictions.append(predicted_subreddit)
    
    test_data['predicted'] = predictions
    return test_data

# Make predictions on the test set
test_data = predict_on_test_set(test_data)

print(test_data[['post', 'predicted']])


# In[3]:


from sklearn.metrics import accuracy_score

test_data['predicted'] = test_data['post'].apply(predict_behavior)

# Calculate accuracy
accuracy = accuracy_score(test_data['labels'], label_encoder.transform(test_data['predicted']))
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




