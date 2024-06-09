import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

data = pd.read_csv(r'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\preprocessed_data.csv')

related_with_race_model = BertForSequenceClassification.from_pretrained(r'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\related_with_race_model_fold_2')

sentiment_model = BertForSequenceClassification.from_pretrained(r'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\sentiment_model_fold_5')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

negative_comments = []

for index, row in data.iterrows():
    comments = eval(row['comments'])
    for comment in comments:
        comment = comment.strip('\n')

        inputs_race = tokenizer(comment, padding=True, truncation=True, return_tensors='pt')
        outputs_race = related_with_race_model(**inputs_race)
        prediction_race = torch.argmax(outputs_race.logits).item()

        if prediction_race == 1:
            inputs_sentiment = tokenizer(comment, padding=True, truncation=True, return_tensors='pt')
            outputs_sentiment = sentiment_model(**inputs_sentiment)
            prediction_sentiment = torch.argmax(outputs_sentiment.logits).item()

            if prediction_sentiment == 1:
                negative_comments.append(comment)

with open('negative_comments.txt', 'w', encoding='utf-8') as file:
    for comment in negative_comments:
        file.write(comment + '\n')
