from google.cloud import storage
import pandas as pd
from textblob import TextBlob
import psycopg2
import os
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# Visualization functions
def visualize_sentiment(df):
    if 'timestamp' in df.columns and 'sentiment' in df.columns and 'ownerFullName' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(12, 6))
        unique_owners = df['ownerFullName'].unique()

        for owner in unique_owners:
            owner_data = df[df['ownerFullName'] == owner]
            plt.plot(owner_data['timestamp'], owner_data['sentiment'], marker='o', linestyle='-', label=owner)

        plt.title('Sentiment Analysis Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        plt.legend(title="Owner Names") 
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        print("Timestamp, sentiment, or ownerFullName column not found in DataFrame.")

def visualize_bar_plot(df):
    if 'ownerFullName' in df.columns and 'sentiment' in df.columns:
        avg_sentiment = df.groupby('ownerFullName')['sentiment'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.bar(avg_sentiment['ownerFullName'], avg_sentiment['sentiment'], color='skyblue')
        plt.xlabel('Owner Names')
        plt.ylabel('Average Sentiment Score')
        plt.title('Average Sentiment Score by Owner')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
    else:
        print("OwnerFullName or sentiment column not found in DataFrame.")

def visualize_box_plot(df):
    if 'ownerFullName' in df.columns and 'sentiment' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.boxplot([df[df['ownerFullName'] == owner]['sentiment'] for owner in df['ownerFullName'].unique()],
                    labels=df['ownerFullName'].unique())
        plt.xlabel('Owner Names')
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment Score Distribution by Owner')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        print("OwnerFullName or sentiment column not found in DataFrame.")

# Google Cloud Storage setup
client = storage.Client(project=os.getenv('PROJECT_ID'))
bucket_name = os.getenv('BUCKET_NAME')
file_name = os.getenv('FILE_NAME')

bucket = client.get_bucket(bucket_name)
blob = bucket.blob(file_name)
blob.download_to_filename(file_name)

if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
    data = []

    try:
        with open(file_name, 'r') as f:
            content = f.read()
            print("File content:\n", content)  

            try:
                data = json.loads(content)  
            except json.JSONDecodeError:
                f.seek(0)  
                for line in f:
                    line = line.strip()  
                    if line:  
                        try:
                            data.append(json.loads(line))  
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON line: {e}")

        if data:  
            df = pd.json_normalize(data)  
            print(df.head())  
            print(df.info())  
        else:
            print("No valid JSON data found.")
            df = None  

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' does not exist.")
        df = None  
    except Exception as e:
        print(f"An error occurred: {e}")
        df = None  

else:
    print("The JSON file does not exist or is empty.")
    df = None  

if df is not None and 'caption' in df.columns:  
    def get_sentiment(text):
        if text: 
            analysis = TextBlob(text)
            return analysis.sentiment.polarity  
        return None  

    df['sentiment'] = df['caption'].apply(get_sentiment)
    print(df[['ownerFullName', 'caption', 'sentiment']])  

    conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host='localhost'
    )
    
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS instagram_sentiment (
        post_id VARCHAR PRIMARY KEY,
        owner_full_name TEXT,
        caption TEXT,
        sentiment FLOAT
    )
    ''')

 
    for index, row in df.iterrows():
        cur.execute('''
        INSERT INTO instagram_sentiment (post_id, owner_full_name, caption, sentiment)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING
        ''', (row.get('id'), row.get('ownerFullName'), row.get('caption'), row.get('sentiment')))

    conn.commit()
    cur.close()
    conn.close()
    print("Data saved to PostgreSQL.")

    visualize_sentiment(df)
    visualize_bar_plot(df)
    visualize_box_plot(df)
else:
    print("DataFrame is empty or missing 'caption' column.")
