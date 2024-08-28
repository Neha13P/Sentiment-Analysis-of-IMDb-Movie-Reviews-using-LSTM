#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install kaggle tensorflow')


# In[2]:


import os
import json

from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[7]:


#DATA COLLECTION- KAGGLE API
kaggle_dictionary = json.load(open(r"C:\Users\nehaa\Downloads\kaggle.json"))


# In[8]:


kaggle_dictionary.keys()


# In[9]:


# setup kaggle credentials as environment variables
os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]


# In[10]:


#get_ipython().system('kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews')


# In[11]:


# unzip the dataset file
with ZipFile("imdb-dataset-of-50k-movie-reviews.zip", "r") as zip_ref:
  zip_ref.extractall()


# In[12]:


#Loading the dataset
data = pd.read_csv("IMDB Dataset.csv")


# In[13]:


data.shape


# In[14]:


data.head()


# In[15]:


data.tail()


# In[16]:


data["sentiment"].value_counts()


# In[18]:


import pandas as pd

# Set the pandas option to opt-in to the future behavior
pd.set_option('future.no_silent_downcasting', True)

# Sample DataFrame
'''data = pd.DataFrame({
    'sentiment': ['positive', 'negative', 'positive', 'negative']
})'''

# Replace values
data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

print(data)


# In[19]:


data["sentiment"].value_counts()


# In[20]:


# split data into training data and test data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# In[21]:


print(train_data.shape)
print(test_data.shape)


# In[22]:


#DATA PREPROCESSING


# In[23]:


# Tokenize text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)


# In[24]:


print(X_train)


# In[25]:


print(X_test)


# In[26]:


Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]


# In[27]:


print(Y_train)


# In[28]:


#LSTM-LONG SHORT-TERM MEMORY


# In[30]:


# build the model

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))


# In[32]:


model.build(input_shape=(None, 200))


# In[33]:


model.summary()


# In[34]:


# compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[35]:


#TRAINING THE MODEL


# In[36]:


model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)
model.save(sa_model.h5)


# In[37]:


#MODEL EVALUATION
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# In[38]:


#BUILDING A PREDICTIVE SYSTEM


# In[39]:


def predict_sentiment(review):
  # tokenize and pad the review
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment


# In[40]:


# example usage
new_review = "This movie was fantastic. I loved it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")


# In[41]:


# example usage
new_review = "This movie was not that good"
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")


# In[42]:


# example usage
new_review = "This movie was ok but not that good."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")


# In[43]:


print(f"The sentiment of the review is: {predict_sentiment(input())}")


# In[44]:


import ipywidgets as widgets
from IPython.display import display


# In[45]:


# Creating text input widget
review_input = widgets.Text(
    value='',
    placeholder='Type your movie review here',
    description='Review:',
    disabled=False
)


# In[46]:


# Creating button widget
button = widgets.Button(description="Predict Sentiment")


# In[47]:


# Creating output widget to display the result
output = widgets.Output()


# In[48]:


# Defining the function to be called when the button is clicked
def on_button_clicked(b):
    with output:
        output.clear_output()  # Clear previous output
        review = review_input.value
        sentiment = predict_sentiment(review)
        print(f"The sentiment of the review is: {sentiment}")


# In[49]:


# Linking the button click event to the function
button.on_click(on_button_clicked)


# In[50]:


# Displaying the widgets
display(review_input, button, output)


# In[ ]:


