import numpy as np
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# loading the Model
model = load_model('next_word_lstm.h5')

with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)

# prediction function
def predict_word(model,tokenizer,text,max_seq_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if(len(token_list) >= max_seq_len):
        token_list=token_list[-(max_seq_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')
    predicted_word=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted_word,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Predcition")
input=st.text_input("Enter a sequence of words","To be or not to")
if st.button("Predict"):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_word(model,tokenizer,input,max_sequence_len)
    st.write(f'Next word: {next_word}')