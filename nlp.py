# CODE REFERENCE: JCharisTech

# core pckgs
from turtle import color
import streamlit as st
import altair as alt

# EDA pckgs
import pandas as pd
import numpy as np

# utils
import joblib

pipe_lr = joblib.load(open("sentimen_classifier.pkl","rb"))

# Fxn
def predict_violence(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    menu = ["Home"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Tingkat Kekerasan Dalam Teks")

        with st.form(key='violence_clf_form'):
            raw_text = st.text_area('Type here (ex: "teman saya berkata kasar" atau "teman saya memukul orang")')
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:
            col1, col2 = st.beta_columns(2)

            # apply fxn here
            prediction = predict_violence(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)

                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                st.write(proba_df)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["tingkat kekerasan","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='tingkat kekerasan', y='probability', color='tingkat kekerasan')
                st.altair_chart(fig, use_container_width=True)




    elif choice == "Monitor":
        st.subheader("Monitor App")
    
    else:
        st.subheader("About")



if __name__ == '__main__':
    main()