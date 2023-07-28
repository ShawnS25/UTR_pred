import streamlit as st
from Bio import SeqIO
from Predictor import predict_file, predict_raw

st.title("5' UTR prediction")

st.subheader("Input sequence")
#x = st.slider('Select a value')
# seq = ""
seq = st.text_input("Input your sequence here", value="")
st.subheader("Upload sequence file")
uploaded = st.file_uploader("Sequence file in FASTA format")
# if uploaded:
    # predict_file(uploaded)
    # seq = SeqIO.read(uploaded, "fasta").seq
st.subheader("Prediction result:")
if st.button("Predict"):
    if uploaded:
        predict_file(uploaded)
    else:
        predict_raw(seq)
    # st.write("Sequence length = ", len(seq))