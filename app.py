import pandas as pd
import re
from transformers import pipeline
from textblob import TextBlob

sentiment_pipeline = pipeline('sentiment-analysis')



# Define a function to preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Remove any characters that are not letters or spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(review):
    result = sentiment_pipeline(review)[0]
    return result['label'], result['score']

def sentiment_analysis_using_transformers():
    df = pd.read_csv('cleaned_reviews.csv')
    df['sentiment'], df['confidence'] = zip(*df['cleaned_review'].apply(get_sentiment))

# Save the DataFrame with sentiment scores to a new CSV file
    df.to_csv('reviews_with_sentiment_transformers.csv', index=False)

def classify_sentiment_textblob(review):
        blob = TextBlob(review)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return 'positive'
        elif polarity < 0:
            return 'negative'
        else:
            return 'neutral'

def sentiment_analysis_using_textblob():
    df = pd.read_csv('cleaned_reviews.csv')

    # Apply the sentiment classification to each review
    df['sentiment'] = df['cleaned_review'].apply(classify_sentiment_textblob)

    # Save the DataFrame with sentiment classifications to a new CSV file
    df.to_csv('reviews_with_sentiment_textblob.csv', index=False)


#Task 1:Load and Preprocess the Data
df = pd.read_excel('user_review.xls')

df.to_csv('user_review.csv', index=False)

# Remove any rows with null values
df = df.dropna()

#Selecting review column
df = df[['review']]
# Apply the preprocessing 
df['cleaned_review'] = df['review'].apply(preprocess_text)
# Save the cleaned data to a new CSV file
cleaned_reviews =  df[['cleaned_review']]
cleaned_reviews.to_csv('cleaned_reviews.csv')

#Task 2:Perform Sentiment Analysis
#Performing sentiment analysis using Transformers
sentiment_analysis_using_transformers()
#to use TextBlob  uncomment the below line
# sentiment_analysis_using_textblob()
print("Sentiment analysis completed and saved to 'reviews_with_sentiment_transformers.csv'.")