import streamlit as st

# File uploader for any file (e.g., PDF)
uploaded_file = st.file_uploader('Upload your PDF file', type='pdf')

if uploaded_file is not None:
    # Get the file name
    file_name = uploaded_file.name
    f_name = file_name.split('.')[0]
    # Display the file name
    st.write(f'Uploaded file name: {f_name}')
