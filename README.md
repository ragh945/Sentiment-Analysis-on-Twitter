# Sentiment Analysis on Virat Kohli Tweets
- This project performs sentiment analysis on tweets about Virat Kohli using a logistic regression model. The tweets are scraped from Twitter using ntscrapper, and the analysis is presented via a Streamlit web app. The project encompasses text preprocessing, vectorization using CountVectorizer, and model serialization using pickle. Additionally, the analysis includes visualizations like word clouds for negative tweets.

- Check the Deployment here https://sentiment-analysis-on-twitter-wbp9uwkt5zitcqjp3thtuy.streamlit.app/
## Project Structure
- Data Collection: Tweets are scraped from Twitter using ntscrapper.
- Text Preprocessing: The tweets undergo preprocessing steps to clean and prepare the text for analysis.
- Text Vectorization: Preprocessed tweets are converted into numerical features using CountVectorizer.
- Model Training: A logistic regression model is trained on the vectorized text data.
- Model Serialization: The trained model and vectorizer are saved using pickle.
- Visualization: Word clouds and other visualizations are generated to represent the data.
  
## Technologies Used
- Python
- pandas
- matplotlib
- seaborn
- scikit-learn
- Streamlit
- wordcloud
- PIL (Python Imaging Library)
- numpy

## Usage
### Load the Pre-trained Model and Vectorizer:
- The logistic regression model and CountVectorizer are loaded from their respective pickle files.

### Input a Tweet:
- Enter a tweet about Virat Kohli in the text input box provided in the Streamlit app.

### Predict Sentiment:
- The app predicts whether the input tweet has a positive, negative, or neutral sentiment and displays the corresponding image.

### Word Cloud Visualization:
- The app generates and displays word clouds for negative tweets using custom shapes and masks.

## Visualization Examples
## 1.Sentiment Prediction
- Positive Sentiment:
- Negative Sentiment:

## 2.Word Clouds
- Custom Shape Word Cloud:
- Grayscale Inverted Word Cloud:
- Default Word Cloud:

![image](https://github.com/user-attachments/assets/638a5b75-f34b-49b0-a4c9-c80efe1829af)

# Conclusion
This project successfully demonstrates the application of sentiment analysis on tweets about Virat Kohli using a logistic regression model and various Python libraries. By leveraging ntscrapper for data collection, CountVectorizer for text vectorization, and Streamlit for deploying the model, we have created an interactive web app that can predict tweet sentiments and visualize the results through word clouds.

## Key takeaways include:
- Data Collection: Efficiently scraping and cleaning tweet data using ntscrapper.
- Text Processing and Model Training: Utilizing CountVectorizer and logistic regression to build a robust sentiment analysis model.
- Interactive Visualization: Employing Streamlit for user interaction and wordcloud for visualizing negative sentiment patterns.
  







