# AI Microbiome Classifier
# author: 박보현

import streamlit as st
import numpy as np
import tensorflow as tf
import json
import re
import pandas as pd

# 모델 로드
model = tf.keras.models.load_model("bh_cnn_rnn_model.h5")

# 서열을 숫자로 변환하는 함수
def convert_sequence_to_numbers(sequence):
    mapping = {"A": 1, "T": 2, "G": 3, "C": 4}
    return [mapping.get(base, 0) for base in sequence]

# 서열을 벡터화하고 padding하는 함수
def vectorize_and_pad_sequences(forward_seq, reverse_seq):
    forward_vec = convert_sequence_to_numbers(forward_seq)
    reverse_vec = convert_sequence_to_numbers(reverse_seq)
    forward_padded = tf.keras.preprocessing.sequence.pad_sequences([forward_vec], maxlen=298, padding="post")
    reverse_padded = tf.keras.preprocessing.sequence.pad_sequences([reverse_vec], maxlen=281, padding="post")
    return np.concatenate([forward_padded, reverse_padded], axis=-1)

# 서열 유효성 검사 함수
def is_valid_sequence(sequence):
    return bool(re.match("^[ATGC]+$", sequence))

# taxonomy label 이름 로딩
with open("taxonomy_labels.json", "r") as f:
    taxonomy_labels = json.load(f)

# Streamlit UI 설정
st.title("AI Microbiome Classifier")

# 라디오 버튼을 통해 입력 옵션 선택
input_option = st.radio("Select Input Method", ("Direct Input", "Upload Files"), horizontal=True)

# Direct Input
if input_option == "Direct Input":
    st.subheader("Direct Sequence Input")

    # Forward와 Reverse sequence 입력 필드
    forward_input = st.text_input("Forward Sequence", key="forward_input", help="Example Forward Sequence: ATGCGTACGTAGCTAGC")
    reverse_input = st.text_input("Reverse Sequence", key="reverse_input", help="Example Reverse Sequence: TGCATCGTAGCATGCA")

    # Run 버튼
    if st.button("Run"):
        if forward_input and reverse_input:
            if not is_valid_sequence(forward_input) or not is_valid_sequence(reverse_input):
                st.write("Please enter valid forward and reverse sequences containing only A, T, G, or C characters.")
            else:
                input_data = vectorize_and_pad_sequences(forward_input, reverse_input)
                input_data = input_data.reshape(1, input_data.shape[1], 1)
                predictions = model.predict(input_data)
                predicted_label_idx = np.argmax(predictions, axis=1)[0]
                predicted_label = taxonomy_labels.get(str(predicted_label_idx), "Unknown label")
                st.markdown(f"Taxonomy Name: *{predicted_label}*")

# Upload Files
elif input_option == "Upload Files":
    st.subheader("Upload Sequence Files")

    # Forward와 Reverse 파일 업로드
    forward_file = st.file_uploader("Upload Forward Sequence File", type="txt", key="forward_file", help="Example content (each comma represents a newline): @header1, ATGCGTACGTAGCTAGC, +, M@KKMLIJIMLMIMLLA in a .txt file")
    reverse_file = st.file_uploader("Upload Reverse Sequence File", type="txt", key="reverse_file", help="Example content (each comma represents a newline): @header1, TACGATCGTACGATCGA, +, GG0?JCG?JB@A><@G; in a .txt file")


    # Run 버튼
    if st.button("Run"):
        if forward_file and reverse_file:
            forward_sequences = forward_file.read().decode("utf-8").splitlines()
            reverse_sequences = reverse_file.read().decode("utf-8").splitlines()
            
            headers = []
            predictions = []

            if len(forward_sequences) < 2 or len(reverse_sequences) < 2:
                st.write("Please ensure both files have sequences in the second line.")
            else:
                headers = []
                predictions = []

                for idx, (forward_seq, reverse_seq) in enumerate(zip(forward_sequences[1::4], reverse_sequences[1::4])):
                    if is_valid_sequence(forward_seq) and is_valid_sequence(reverse_seq):
                        input_data = vectorize_and_pad_sequences(forward_seq, reverse_seq)
                        input_data = input_data.reshape(1, input_data.shape[1], 1)
                        prediction = model.predict(input_data)
                        predicted_label_idx = np.argmax(prediction, axis=1)[0]
                        predicted_label = taxonomy_labels.get(str(predicted_label_idx), "Unknown label")
                        
                        headers.append(forward_sequences[4 * idx].strip())
                        predictions.append(predicted_label)
                    else:
                        st.write(f"Invalid characters in sequence {idx+1}. Please use only A, T, G, C characters.")
                        break

                # 결과 데이터프레임 생성 및 상위 5개 결과 출력
                result_df = pd.DataFrame({"Header": headers, "Taxonomy Name": predictions})
                st.write("Top 5 classifications from uploaded files:")
                st.write(result_df.head())

                # CSV 파일 다운로드
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download classifications as CSV",
                    data=csv,
                    file_name="taxonomy_classifications.csv",
                    mime="text/csv",
                )
        else:
            st.write("Please upload both forward and reverse sequence files.")
