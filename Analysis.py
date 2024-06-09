import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

data = pd.read_csv(r'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\preprocessed_data.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

related_with_race_model = BertForSequenceClassification.from_pretrained('C:/Users/CharlesQX/Desktop/python/ResearchProjectData/related_with_race_model_fold_2')

sentiment_model = BertForSequenceClassification.from_pretrained('C:/Users/CharlesQX/Desktop/python/ResearchProjectData/sentiment_model_fold_5')

total_comments = 0
race_related_comments = 0
negative_comments = 0
race_related_negative_comments = 0
all_comments = []

for index, row in data.iterrows():
    comments = eval(row['comments']) 
    views = row['Views']
    total_comments += len(comments)
    race_related_count = 0
    negative_count = 0
    sentiment_predictions = []
    
    for comment in comments:
        comment = comment.strip('\n')
        
        inputs = tokenizer(comment, padding=True, truncation=True, return_tensors='pt')
        outputs = related_with_race_model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
        
        if prediction == 1:
            sentiment_inputs = tokenizer(comment, padding=True, truncation=True, return_tensors='pt')
            sentiment_outputs = sentiment_model(**sentiment_inputs)
            sentiment_prediction = torch.argmax(sentiment_outputs.logits).item()
            sentiment_predictions.append(sentiment_prediction)
            if sentiment_prediction == 1:
                negative_count += 1
        
        race_related_count += prediction
        
    race_related_comments += race_related_count
    negative_comments += negative_count
    race_related_negative_comments += negative_count if race_related_count > 0 else 0
    
    percentage_negative_comments = (negative_count / len(comments)) * 100 if len(comments) > 0 else 0
    
    correlation = views * percentage_negative_comments
    
    all_comments.extend([[comment, 'positive/neutral', 'negative'][sentiment_prediction] for sentiment_prediction in sentiment_predictions])
    
percentage_race_related = (race_related_comments / total_comments) * 100
percentage_negative = (negative_comments / total_comments) * 100
percentage_race_related_negative = (race_related_negative_comments / race_related_comments) * 100 if race_related_comments > 0 else 0

print("Total Number of Comments:", total_comments)
print("Total Number of Race Related Comments:", race_related_comments, "(", percentage_race_related, "%)")
print("Total Number of Negative Comments:", negative_comments, "(", percentage_negative, "%)")
print("Percentage of Negative Comments Among Race Related Comments:", percentage_race_related_negative, "%")

comments_df = pd.DataFrame(all_comments, columns=['Comments', 'Sentiment'])
comments_df.to_csv('summarized_comments.csv', index=False)