import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.optim as optim

data = pd.read_csv(r'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\Training Data.csv')

data = data[data['Sentiment'] != -1]

data['Sentiment'] = data['Sentiment'].map({0: 0, 1: 1})

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

sentiment_accuracies = []

for fold, (train_index, test_index) in enumerate(skf.split(data['comments'], data['Sentiment'])):
    print(f"Fold {fold + 1}/{k_folds}")
    
    train_data, test_data = data.iloc[train_index], data.iloc[test_index]
    
    train_encodings = tokenizer(list(train_data['comments']), padding=True, truncation=True, return_tensors='pt')
    test_encodings = tokenizer(list(test_data['comments']), padding=True, truncation=True, return_tensors='pt')
    
    train_labels_sentiment = torch.tensor(list(train_data['Sentiment']))
    test_labels_sentiment = torch.tensor(list(test_data['Sentiment']))
    
    sentiment_model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    sentiment_model.train()
    
    optimizer = optim.AdamW(sentiment_model.parameters(), lr=2e-5)
    for epoch in range(3):
        running_loss = 0.0
        for i in range(0, len(train_labels_sentiment), 32):
            optimizer.zero_grad()
            outputs = sentiment_model(**train_encodings[i:i+32], labels=train_labels_sentiment[i:i+32])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}: Training Loss {running_loss / (len(train_labels_sentiment) / 32)}")
    
    with torch.no_grad():
        sentiment_logits = sentiment_model(**test_encodings).logits
        sentiment_predictions = torch.argmax(sentiment_logits, dim=1)
        sentiment_accuracy = accuracy_score(test_labels_sentiment, sentiment_predictions)
        sentiment_accuracies.append(sentiment_accuracy)
        print(f"Sentiment Accuracy: {sentiment_accuracy}")
    
    sentiment_model_save_path = fr'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\sentiment_model_fold_{fold + 1}'
    sentiment_model.save_pretrained(sentiment_model_save_path)
    print(f"Saved Sentiment model for fold {fold + 1} at {sentiment_model_save_path}")
    
    print(f"Average Sentiment Accuracy: {sum(sentiment_accuracies) / (fold + 1)}")