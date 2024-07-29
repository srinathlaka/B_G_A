import streamlit as st
import pandas as pd

# Upload file
def upload_file():
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
    return uploaded_file

# Read data
@st.cache_data
def read_data(uploaded_file, rows, columns):
    if uploaded_file is not None:
        try:
            st.success("File uploaded successfully!")
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, header=None)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=None)
            df = adjust_dataframe_layout(df, rows, columns)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    return None

# Adjust DataFrame layout to match specified rows and columns
def adjust_dataframe_layout(df, rows, columns):
    total_columns = rows * columns
    labels = generate_labels(rows, columns)
    
    if df.shape[1] < total_columns + 1:
        extra_cols = pd.DataFrame(np.nan, index=df.index, columns=range(df.shape[1], total_columns + 1))
        df = pd.concat([df, extra_cols], axis=1)
    
    df = df.iloc[:, :total_columns + 1]
    df.columns = ["Time"] + labels
    return df

# Generate labels for well selection
def generate_labels(rows, cols):
    labels = []
    for i in range(rows):
        for j in range(cols):
            label = chr(ord('A') + i) + str(j + 1)
            labels.append(label)
    return labels

# Select layout for wells
def select_layout():
    layout_option = st.selectbox("Select layout option", ["Select from presets", "Custom"])
    if layout_option == "Select from presets":
        well_format = st.selectbox("Select well format", ["24 well rows 4 column 6", "96 well rows 8 column 12",
                                                         "84 well rows 7 column 12", "1536 well rows 32 column 48"])
        if well_format == "24 well rows 4 column 6":
            return 4, 6
        elif well_format == "96 well rows 8 column 12":
            return 8, 12
        elif well_format == "84 well rows 7 column 12":
            return 7, 12
        elif well_format == "1536 well rows 32 column 48":
            return 32, 48
    else:
        custom_rows = st.number_input("Enter number of rows", min_value=1, step=1)
        custom_columns = st.number_input("Enter number of columns", min_value=1, step=1)
        return int(custom_rows), int(custom_columns)
