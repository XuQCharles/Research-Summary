import pandas as pd
import matplotlib.pyplot as plt
file_path = r'C:\Users\CharlesQX\Desktop\python\ResearchProjectData\merged_data.csv'
data = pd.read_csv(file_path)

data['Publication Date'] = pd.to_datetime(data['Publication Date'])
data['Month'] = data['Publication Date'].dt.to_period('M')
grouped_data = data.groupby('Month').agg({'Number of Negative Comments': 'sum', 'Number of Danmu': 'sum'})
grouped_data['Percentage of Negative Comments'] = (grouped_data['Number of Negative Comments'] / grouped_data['Number of Danmu']) * 100

plt.figure(figsize=(10, 6))
plt.plot(grouped_data.index.to_timestamp(), grouped_data['Percentage of Negative Comments'], marker='o', linestyle='-')
plt.title('Percentage of Negative Comments Over Time (Monthly)')
plt.xlabel('Publication Date')
plt.ylabel('Percentage of Negative Comments')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()
