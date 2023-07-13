
from keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
import random
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import data_utils
from keras.preprocessing.text import Tokenizer
import pandas as pd


# import dataset from csv file
df = pd.read_csv('tweet_sentiment_450K.csv')


review_df = df[['text', 'airline_sentiment']]
review_df = review_df[review_df['airline_sentiment'] != 'neutral']
review_df = review_df.sample(frac=1)  # Shuffle array
print(review_df.shape)
review_df.head(5)


print(review_df["airline_sentiment"].value_counts())


sentiment_label = review_df.airline_sentiment.factorize()
tweet = review_df.text.values

print(sentiment_label)


tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(tweet)
print(len(tokenizer.word_index))  # dictionary of words and their index
encoded_docs = tokenizer.texts_to_sequences(tweet)


padded_sequence = data_utils.pad_sequences(
    tokenizer.texts_to_sequences(tweet), maxlen=200)


vocab_size = len(tokenizer.word_index) + 1
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())


history = model.fit(
    padded_sequence, sentiment_label[0], validation_split=0.1, epochs=4, batch_size=16)


plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


def predict_sentiment(text: str):
    tw = tokenizer.texts_to_sequences([text])
    tw = data_utils.pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    return sentiment_label[1][prediction]


test_data_frame = pd.read_csv('tweet_sentiment.csv')
text = test_data_frame["text"]
sentiment = test_data_frame['airline_sentiment']


total = 100
score = 0

for i in range(total):
    index = int(random.random()*len(text))
    test_text = text[index]
    result = predict_sentiment(test_text)
    score += 1

    print(i)
    print(result)


# Print score

print(f"Predicted with accuracy of {(score / total)*100}")


# Save the model

model.save("./sa_450K")


for i in range(2):
    user_input = input("Enter text to analyze: ")
    print(predict_sentiment(user_input))
