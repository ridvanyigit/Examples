# streamli.io
# in terminal : conda activate base
# pip install -U streamlit
# pip install -U plotly

### You can run your app with : streamlit run app.py

import streamlit as st
import pickle 

# Loading the trained Model

model = pickle.load(open('spam_classifier.pkl', 'rb'))

# Create a Title
st.title('Predicting if Message is Spam or Not')

message = st.text_input('Enter a Message')

submit = st.button('Predict')

if submit:
    prediction = model.predict([message])

    #print(prediction)
    #st.write(prediction)

    if prediction[0]=='spam':
        st.warning('This Message is Spam!')

    else:
        st.success('This Message is Legit (Ham)')

st.balloons()

