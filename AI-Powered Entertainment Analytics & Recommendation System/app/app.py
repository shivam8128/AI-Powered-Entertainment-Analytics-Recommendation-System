import streamlit as st

st.title("Entertainment AI System")

st.header("Movie Recommendation")
user_id = st.number_input("Enter User ID", min_value=1)

if st.button("Recommend"):
    st.write("Top recommendations coming soon...")

st.header("Sentiment Analysis")
text = st.text_area("Enter Review")

if st.button("Analyze"):
    st.write("Sentiment: Positive")
