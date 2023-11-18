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

def parse_time_duration(time_str):
    # Split the time string into hours, minutes, and seconds
    h, m, s = time_str.split(':')
    return datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))



def read_bvp_data(uploaded_file):
    # Read and ignore the first two lines (metadata)
    initial_timestamp = float(uploaded_file.readline().decode().strip().split(',')[0])
    sample_rate = float(uploaded_file.readline().decode().strip().split(',')[0])

    # Read the BVP data into a DataFrame
    bvp_data = pd.read_csv(uploaded_file, header=None)
    bvp_data['Timestamp'] = pd.to_datetime(initial_timestamp, unit='s') + pd.to_timedelta(bvp_data.index / sample_rate, unit='s')

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

def find_closest_time(event_time_delta, ibi_data):
    # Convert Timedelta to total seconds
    event_time_seconds = event_time_delta.total_seconds()

    # Ensure 'Elapsed Time' is in a numeric format
    ibi_data['Elapsed Time'] = pd.to_numeric(ibi_data['Elapsed Time'], errors='coerce')

    # Drop NA values before sorting
    ibi_data = ibi_data.dropna(subset=['Elapsed Time'])

    # Find the closest time in ibi_data
    closest_time = ibi_data.iloc[(ibi_data['Elapsed Time'] - event_time_seconds).abs().argsort()[:1]]
    return closest_time


def process_bvp_signal_and_compute_hrv(bvp_data, sampling_rate):
    # Assuming bvp_data is a one-column DataFrame with the BVP signal
    bvp_signal = bvp_data.iloc[:, 0]

    # Clean the BVP signal
    processed_signal = nk.ppg_clean(bvp_signal, sampling_rate=sampling_rate)

    # Find R-peaks in the processed BVP signal
    r_peaks_info = nk.ppg_findpeaks(processed_signal)
    r_peaks = r_peaks_info['PPG_Peaks']  # This should be a list or Series of R-peak indices

    # Compute HRV metrics
    hrv_metrics = nk.hrv(r_peaks, sampling_rate=sampling_rate, show=True)
    return hrv_metrics, processed_signal




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

    # Convert the timestamp to relative time (since start of the recording)
    # Apply this function to the 'Elapsed Time' column
    tags_data['Relative Time'] = tags_data['Elapsed Time'].apply(parse_time_duration)
    # User selects start and end tags for each event
    event_tags = tags_data['Relative Time'].tolist()
    start_tag = st.selectbox('Select Start Tag', event_tags, key='start_tag')
    end_tag = st.selectbox('Select End Tag', event_tags, key='end_tag')

    # Find the corresponding time in data for start and end tags
    closest_start_time = find_closest_time(pd.to_timedelta(start_tag), ibi_data)
    closest_end_time = find_closest_time(pd.to_timedelta(end_tag), ibi_data)

    # Process the segment between the selected start and end times
    segment = bvp_data[(bvp_data['Timestamp'] >= closest_start_time) & (bvp_data['Timestamp'] <= closest_end_time)]
    hrv_metrics, processed_segment = process_bvp_signal_and_compute_hrv(segment, 64)
    st.write(hrv_metrics)

    # Visualization (if needed)
    st.line_chart(processed_segment)

    # Download results
    st.download_button(label="Download HRV Metrics as CSV", data=hrv_metrics.to_csv(), file_name='hrv_metrics.csv', mime='text/csv')
