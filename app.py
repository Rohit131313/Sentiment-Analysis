import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('stopwords')

#loading models
model = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def stopwordss():
    removed = []
    stopword = stopwords.words('english')
    for i in stopword:
        if "n't" in i:
            removed.append(i)
        if i == "not":
            removed.append(i)
    for i in removed:
        stopword.remove(i)
    return stopword



def preprocess(content,stopword):
    # sentence = nltk.sent_tokenize(i)
    sentence = re.sub(r'[^a-zA-Z\']',' ',content)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    sentence = sentence.lower()
    # sentence = nltk.word_tokenize(sentence)
    sentence = sentence.split()
    sentence = [lemmatizer.lemmatize(word) for word in sentence if not word in stopword]
    sentence = ' '.join(sentence)
    return sentence

def main():
    st.title("Sentimental Analysis")

    # Input from user
    user_input = st.text_input("Enter text here")

    if user_input:
        stopword = stopwordss()
        sentence = preprocess(user_input,stopword)
        input_feature = tfidf.transform([sentence])
        prediction = model.predict(input_feature)

        # Display the prediction
        if prediction == 4:
            prediction = "Positive"
        elif prediction == 0:
            prediction = "Negative"
        st.write("Predicted sentiment:", prediction)
    else:
        st.write("Please enter some text.")

if __name__ == "__main__":
    main()