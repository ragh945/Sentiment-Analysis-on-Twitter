import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image, ImageOps
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS


Inno = Image.open("Inno (2).jpeg")
st.image(Inno)
# Correct file paths for images
vk = Image.open("vh_tw2227900.jpeg")
vk_array = np.array(vk)

# Display the image using matplotlib to avoid the error
fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(vk_array, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig,use_container_width=True)

# Display the image using Streamlit's image function
#st.image(vk, use_column_width=False)
st.title("Sentiment Analysis on Virat Kohli Tweets")

# Load the CSV file
df = pd.read_csv("VK_final.csv")



# Load the pre-trained model and vectorizer
model_path = "Tweets.pkl"
vectorizer_path = "Bagw_vectorization.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    Bagw = pickle.load(vectorizer_file)

# Input review from user
comment = st.text_input("Enter your Tweet:")

# Transform the input review using the loaded vectorizer
if st.button("Submit"):
    data = Bagw.transform([comment]).toarray()
    pred = model.predict(data)[0]
    pos=Image.open("Positive.jpg")
    neg=Image.open("Negative.jpg")
    if pred=="Negative":
        st.image(neg)
    elif pred=="Positive":
        st.image(pos)
    else:
        st.write("Neutral")
         
    

    #st.write(pred)

# Generate word cloud for negative tweets
negative_tweet = df.loc[df["Opinion"] == "Negative", "Text"]
negative_words = " ".join(negative_tweet.values)

# Load the images
image_path1 = "cr.png"  # Path to first image
image_path2 = "Vk_sil.png"  # Path to second image

meta_mask1 = np.array(Image.open(image_path1))
image2 = Image.open(image_path2)
image2 = ImageOps.grayscale(image2)  # Convert image to grayscale
image2 = ImageOps.invert(image2)  # Invert image to have the shape as white and background as black
meta_mask2 = np.array(image2)

# Create the word cloud for negative tweets
wordcloud1 = WordCloud(
    background_color='white',
    mask=meta_mask1,
    contour_width=3,
    contour_color='black',
    width=800, height=300,
).generate(negative_words)

# Create the second word cloud
wordcloud2 = WordCloud(
    background_color='black',
    mask=meta_mask2,
    contour_width=0.3,
    contour_color='white',
    width=800, height=300,
).generate(negative_words)

# Load the third image to display
worldcloud3_image = Image.open("Vk_sil.png")
worldcloud3_array = np.array(worldcloud3_image)

# Display the word clouds using subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 10))

axes[0].imshow(wordcloud1, interpolation='bilinear')
axes[0].axis('off')  # Remove axes

axes[1].imshow(wordcloud2, interpolation='bilinear')
axes[1].axis('off')  # Remove axes

axes[2].imshow(worldcloud3_array, interpolation='bilinear')
axes[2].axis('off')  # Remove axes

# Display the word clouds in Streamlit
st.pyplot(fig)


