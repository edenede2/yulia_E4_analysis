import streamlit as st
import pandas as pd
import numpy as np
import datetime
import neurokit2 as nk

def read_and_convert_data(uploaded_file, file_type):
    # Read the initial timestamp and sample rate from the file's first two lines
    initial_timestamp_line = uploaded_file.readline().decode().strip()
    initial_timestamp_parts = initial_timestamp_line.split(',')
    initial_timestamp = float(initial_timestamp_parts[0].strip())

    if file_type == 'BVP':
        # BVP specific processing
        sample_rate_line = uploaded_file.readline().decode().strip()
        sample_rate = float(sample_rate_line.split(',')[0].strip())
        df = pd.read_csv(uploaded_file, header=None)
        df['Timestamp'] = pd.to_datetime(initial_timestamp, unit='s') + pd.to_timedelta(df.index / sample_rate, unit='s')
    elif file_type == 'IBI':
        # IBI specific processing
        df = pd.read_csv(uploaded_file, skiprows=1, header=None)
        df.rename(columns={1: 'IBI'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(initial_timestamp, unit='s') + pd.to_timedelta(df['IBI'], unit='s')
    else:
        # For other files, directly read into DataFrame assuming timestamps are in the first column
        df = pd.read_csv(uploaded_file, header=None)
        df['Timestamp'] = pd.to_datetime(df[0], unit='s')


    # Calculate elapsed time from the reference start time
    reference_start_time = pd.to_datetime(initial_timestamp, unit='s')
    df['Elapsed Time'] = (df['Timestamp'] - reference_start_time).dt.total_seconds()
    df['Elapsed Time'] = df['Elapsed Time'].apply(lambda x: str(datetime.timedelta(seconds=int(x))))

    return df
    
def read_bvp_data(uploaded_file):
    # Read and ignore the first two lines (metadata)
    uploaded_file.readline()  # Ignore first line
    uploaded_file.readline()  # Ignore second line

    # Read the BVP data into a DataFrame
    bvp_data = pd.read_csv(uploaded_file, header=None)
    if bvp_data.shape[1] != 1:
        raise ValueError("BVP data should be a single column. Found: {} columns".format(bvp_data.shape[1]))

    return bvp_data

# Helper Functions
def convert_to_elapsed_time(df, initial_timestamp):
    # Convert initial timestamp to datetime
    initial_time = pd.to_datetime(initial_timestamp, unit='s')
    
    # Calculate time elapsed since initial timestamp for each row
    df['Elapsed Time'] = df.index / df.iloc[1, 0] + initial_timestamp
    df['Elapsed Time'] = (initial_time + pd.to_timedelta(df['Elapsed Time'], unit='s')).dt.strftime('%H:%M:%S:%f')
    return df

def match_event_tags(tags_df, data_df):
    # Assuming tags_df contains Unix timestamps of events
    # Convert these to datetime
    tags_df['Timestamp'] = pd.to_datetime(tags_df[0], unit='s')

    # Find corresponding times in data_df
    matched_events = pd.merge_asof(tags_df, data_df, on='Timestamp')
    return matched_events

def find_closest_time(event_time, ibi_data):
    # Convert event_time to a string in the correct format if it's not already a string
    if not isinstance(event_time, str):
        event_time = event_time.strftime('%H:%M:%S:%f')

    event_datetime = datetime.datetime.strptime(event_time, '%H:%M:%S:%f')

    # Adjusted to use 'Timestamp' for comparison
    closest_time = ibi_data.iloc[(ibi_data['Timestamp'] - event_datetime).abs().argsort()[:1]]
    return closest_time

def process_bvp_signal(bvp_data, sampling_rate):
    # Check if bvp_data is a DataFrame and has only one column
    if isinstance(bvp_data, pd.DataFrame) and bvp_data.shape[1] == 1:
        # Extract the BVP signal as a series
        bvp_signal = bvp_data.iloc[:, 0]
    else:
        raise ValueError("BVP data should be a DataFrame with only one column.")

    # Clean the BVP signal
    processed_signal = nk.ppg_clean(bvp_signal, sampling_rate=sampling_rate)
    return processed_signal



# Streamlit App
st.title('HRV Metrics from PPG Data')

# File Uploaders
bvp_file = st.file_uploader("Upload BVP.csv", type="csv")
tags_file = st.file_uploader("Upload tags.csv", type="csv")
ibi_file = st.file_uploader("Upload IBI.csv", type="csv")

if bvp_file and tags_file and ibi_file:
    bvp_data = read_bvp_data(bvp_file)
    tags_data = read_and_convert_data(tags_file, 'tags')
    ibi_data = read_and_convert_data(ibi_file, 'IBI')

    # User selects an event from the dropdown
    event_choices = tags_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    selected_event = st.selectbox('Select Event', event_choices)

    # Find the corresponding time in data
    selected_event_time = pd.to_datetime(selected_event)
    closest_time = find_closest_time(selected_event_time, ibi_data)

    # Process BVP Signal
    processed_bvp = process_bvp_signal(bvp_data, 64)  # Assuming 64Hz sampling rate

    # Calculate HRV metrics
    hrv_metrics = nk.hrv(processed_bvp, sampling_rate=64, show=True)
    st.write(hrv_metrics)

    # Visualization (if needed)
    st.line_chart(processed_bvp)

    # Download results
    st.download_button(label="Download HRV Metrics as CSV", data=hrv_metrics.to_csv(), file_name='hrv_metrics.csv', mime='text/csv')
