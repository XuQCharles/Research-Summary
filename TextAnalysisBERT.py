import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.optim as optim

data = pd.read_csv(r'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\Training Data.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

related_with_race_accuracies = []

for fold, (train_index, test_index) in enumerate(skf.split(data['comments'], data['Related With Race'])):
    print(f"Fold {fold + 1}/{k_folds}")
    
    train_data, test_data = data.iloc[train_index], data.iloc[test_index]
    
    train_encodings = tokenizer(list(train_data['comments']), padding=True, truncation=True, return_tensors='pt')
    test_encodings = tokenizer(list(test_data['comments']), padding=True, truncation=True, return_tensors='pt')
    
    train_labels_related_with_race = torch.tensor(list(train_data['Related With Race']))
    test_labels_related_with_race = torch.tensor(list(test_data['Related With Race']))
    
    related_with_race_model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    related_with_race_model.train()
    
    learning_rate = 2e-5
    batch_size = 32
    num_epochs = 3
    
    optimizer = optim.AdamW(related_with_race_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, len(train_labels_related_with_race), batch_size):
            optimizer.zero_grad()
            outputs = related_with_race_model(**train_encodings[i:i+batch_size], labels=train_labels_related_with_race[i:i+batch_size])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}: Training Loss {running_loss / (len(train_labels_related_with_race) / batch_size)}")
        scheduler.step()
    
    related_with_race_model_save_path = rf'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\related_with_race_model_fold_{fold + 1}'
    try:
        related_with_race_model.save_pretrained(related_with_race_model_save_path)
    except Exception as e:
        print(f"Error saving Related With Race model for fold {fold + 1}: {str(e)}")
    
    with torch.no_grad():
        related_with_race_logits = related_with_race_model(**test_encodings).logits
        related_with_race_predictions = torch.argmax(related_with_race_logits, dim=1)
        related_with_race_accuracy = accuracy_score(test_labels_related_with_race, related_with_race_predictions)
        related_with_race_accuracies.append(related_with_race_accuracy)
        print(f"Related With Race Accuracy: {related_with_race_accuracy}")

    print(f"Average Related With Race Accuracy: {sum(related_with_race_accuracies) / (fold + 1)}")