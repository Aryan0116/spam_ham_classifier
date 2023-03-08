import streamlit as st 
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords 
ps= PorterStemmer()
def text_processing(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    z=[]
    for i in text:
        if i.isalnum() and i not in string.punctuation and i not in stopwords.words('english'):
            z.append(i)
    text=z[:]
    z.clear()
    for i in text:
        z.append(ps.stem(i))
    return " ".join(z)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model4.pkl','rb'))

st.title("Email/SMS spam Classifier")

input_sms =  st.text_area("Enter the Message")

if st.button('predict'):
    
    transform_sms = text_processing(input_sms)

    vector_input = tfidf.transform([transform_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("spam")
    else:
        st.header("Not spam")