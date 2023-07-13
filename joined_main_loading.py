
from numpy import float32
from tensorflow import Tensor
from keras.models import load_model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import data_utils
import pandas as pd


# import dataset from csv file
df = pd.read_csv('tweet_sentiment_450K_shuffled.csv')


review_df = df[['text', 'airline_sentiment']]
review_df = review_df[review_df['airline_sentiment'] != 'neutral']
print(review_df.shape)
print(review_df.head(5))
print(review_df.tail(5))


print(review_df["airline_sentiment"].value_counts())


sentiment_label = review_df.airline_sentiment.factorize()
tweet = review_df.text.values

print(sentiment_label)


tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(tweet)
tokenizer.word_index  # dictionary of words and their index
encoded_docs = tokenizer.texts_to_sequences(tweet)


loaded_model: Sequential = load_model("./sa_450K")  # type: ignore


new_tokenizer = Tokenizer(num_words=100000)
new_tokenizer.fit_on_texts(tweet)


def predict_sentiment_loaded(text: str):
    tw = new_tokenizer.texts_to_sequences([text])
    tw = data_utils.pad_sequences(tw, maxlen=200)
    tensor: Tensor = loaded_model.__call__(tw, training=True)  # type: ignore
    raw_prediction: float32 = tensor.dtype.as_numpy_dtype(
        tensor.numpy()[0][0])  # type: ignore
    prediction = int(raw_prediction.round().item())
    # Calculate confidence using raw prediction
    confidence = ((raw_prediction.item() if prediction ==
                  1 else 1 - raw_prediction.item())-0.5)*2
    return (sentiment_label[1][prediction], confidence)


# Get and predict sentiment of text
while True:
    user_input = input("Enter text to analyze: ")
    if user_input == "!exit":
        break
    prediction, confidence = predict_sentiment_loaded(user_input)
    print(f"Input: {user_input}")
    print(f"Sentiment: {prediction}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("================================")


# Random Assertions

assert predict_sentiment_loaded("I am so happy")[0] == "positive"
assert predict_sentiment_loaded("I am so sad")[0] == "negative"
try:
    assert predict_sentiment_loaded("I am so angry!")[0] == "negative"
except AssertionError:
    print("errored as expected")
print(predict_sentiment_loaded("I am so angry!"))
