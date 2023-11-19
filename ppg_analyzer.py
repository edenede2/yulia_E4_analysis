import streamlit as st
import pandas as pd
import numpy as np
import datetime
import neurokit2 as nk


# New Function Definitions
def read_initial_timestamp(uploaded_file):
    initial_timestamp_line = uploaded_file.readline().decode().strip()
    initial_timestamp = float(initial_timestamp_line.split(',')[0].strip())
    return initial_timestamp

def read_bvp_data(uploaded_file):
    initial_timestamp = read_initial_timestamp(uploaded_file)
    sample_rate = float(uploaded_file.readline().decode().strip().split(',')[0])
    bvp_data = pd.read_csv(uploaded_file, header=None)
    bvp_data['Timestamp'] = pd.to_datetime(initial_timestamp, unit='s', utc=True) + pd.to_timedelta(bvp_data.index / sample_rate, unit='s')
    return bvp_data, initial_timestamp

def read_ibi_data(uploaded_file):
    initial_timestamp = read_initial_timestamp(uploaded_file)
    ibi_data = pd.read_csv(uploaded_file, skiprows=1, header=None)
    ibi_data.rename(columns={0: 'Relative Time', 1: 'IBI'}, inplace=True)
    ibi_data['Timestamp'] = pd.to_datetime(initial_timestamp, unit='s', utc=True) + pd.to_timedelta(ibi_data['Relative Time'], unit='s')
    return ibi_data, initial_timestamp


def read_tags_data(uploaded_file):
    tags_data = pd.read_csv(uploaded_file, header=None)
    tags_data['Timestamp'] = pd.to_datetime(tags_data[0], unit='s', utc=True)
    initial_timestamp = tags_data.iloc[0, 0]
    initial_timestamp_utc = pd.to_datetime(initial_timestamp, unit='s', utc=True)  # Convert to UTC
    tags_data['Elapsed Time'] = (tags_data['Timestamp'] - initial_timestamp_utc).dt.total_seconds()
    return tags_data, initial_timestamp


def parse_time_duration(time_str):
    # Split the time string into hours, minutes, and seconds
    h, m, s = time_str.split(':')
    return datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))


def find_gaps(ibi_data, threshold=20.0):
    gaps = []
    for i in range(1, len(ibi_data)):
        # Ensure ibi_data is a DataFrame with 'Timestamp' column
        if 'Timestamp' in ibi_data.columns:
            time_diff = (ibi_data.iloc[i]['Timestamp'] - ibi_data.iloc[i - 1]['Timestamp']).total_seconds()
            if time_diff > threshold:
                gaps.append(i)  # Append the index instead of timestamp
        else:
            raise ValueError("Timestamp column not found in ibi_data")
    return gaps


    
def remove_gaps_from_bvp(bvp_data, gaps):
    for index in gaps:
        if 0 < index < len(bvp_data):
            start = bvp_data.iloc[index - 1]['Timestamp']
            end = bvp_data.iloc[index]['Timestamp']
            bvp_data = bvp_data[(bvp_data['Timestamp'] < start) | (bvp_data['Timestamp'] > end)]
    return bvp_data


# Helper Functions
def convert_to_elapsed_time(df, initial_timestamp):
    # Convert initial timestamp to datetime
    initial_time = pd.to_datetime(initial_timestamp, unit='s', utc=True)
    
    # Calculate time elapsed since initial timestamp for each row
    df['Elapsed Time'] = df.index / df.iloc[1, 0] + initial_timestamp
    df['Elapsed Time'] = (initial_time + pd.to_timedelta(df['Elapsed Time'], unit='s')).dt.strftime('%H:%M:%S:%f')
    return df

def match_event_tags(tags_df, data_df):
    # Assuming tags_df contains Unix timestamps of events
    # Convert these to datetime
    tags_df['Timestamp'] = pd.to_datetime(tags_df[0], unit='s', utc=True)

    # Find corresponding times in data_df
    matched_events = pd.merge_asof(tags_df, data_df, on='Timestamp')
    return matched_events

def find_closest_time(event_time_delta, ibi_data, reference_start_time):
    # Convert the string timestamps to total seconds
    ibi_data['Elapsed Seconds'] = pd.to_numeric(ibi_data.iloc[:, 0], errors='coerce')

    # Convert event_time_delta to total seconds
    event_time_seconds = event_time_delta.total_seconds()

    # Find the closest time in ibi_data
    closest_index = (ibi_data['Elapsed Seconds'] - event_time_seconds).abs().idxmin()
    closest_seconds = ibi_data.iloc[closest_index, 0]

    # Convert to actual timestamp in UTC
    closest_timestamp = pd.to_datetime(reference_start_time, unit='s', utc=True) + datetime.timedelta(seconds=closest_seconds)
    return closest_timestamp



def process_and_analyze_bvp(bvp_segment, sampling_rate):
    """
    Clean the BVP signal and compute HRV metrics.
    """
    # Use the first column of the DataFrame
    bvp_signal = bvp_segment.iloc[:, 0]

    # Clean the BVP signal
    cleaned_bvp = nk.ppg_clean(bvp_signal, sampling_rate=sampling_rate)

    # Find R-peaks in the cleaned BVP signal
    peaks_info = nk.ppg_findpeaks(cleaned_bvp, sampling_rate=sampling_rate)
    r_peaks = peaks_info['PPG_Peaks']

    # Compute HRV metrics
    hrv_metrics = nk.hrv(r_peaks, sampling_rate=sampling_rate, show=False)
    return hrv_metrics, cleaned_bvp, r_peaks

def format_time_for_display(timestamp, initial_timestamp):
    elapsed_time = timestamp - initial_timestamp
    return str(datetime.timedelta(seconds=int(elapsed_time.total_seconds())))



# Streamlit App
st.title('HRV Metrics from PPG Data')

# File Uploaders
bvp_file = st.file_uploader("Upload BVP.csv", type="csv")
tags_file = st.file_uploader("Upload tags.csv", type="csv")
ibi_file = st.file_uploader("Upload IBI.csv", type="csv")



if bvp_file and tags_file and ibi_file:
    # Read the data and capture initial_timestamp
    bvp_data, bvp_initial_timestamp = read_bvp_data(bvp_file)
    tags_data, tags_initial_timestamp = read_tags_data(tags_file)
    ibi_data, ibi_initial_timestamp = read_ibi_data(ibi_file)

    # Use any of the initial timestamps (assuming they are the same)
    reference_start_time = pd.to_datetime(bvp_initial_timestamp, unit='s')

    
    # Convert the timestamp to relative time (since start of the recording)
    # Apply this function to the 'Elapsed Time' column
    tags_data['Relative Time'] = tags_data['Elapsed Time'].apply(lambda x: str(datetime.timedelta(seconds=int(x))))
    # User selects start and end tags for each event
    # User selects the event name
    event_names = ["Baseline", "5-Digit test", "Exposure", "Event1", "Event2"]
    selected_event = st.selectbox('Select Event Name', event_names)

    # User selects start and end tags for the chosen event
    event_tags = tags_data['Relative Time'].tolist()
   # Assuming this is part of your Streamlit app
    start_tag = st.selectbox('Select Start Tag', event_tags, key='start_tag')
    end_tag = st.selectbox('Select End Tag', event_tags, key='end_tag')
    
    # Now start_tag and end_tag are defined and can be used
    start_tag_timedelta = pd.to_timedelta(start_tag)
    end_tag_timedelta = pd.to_timedelta(end_tag)
    
    # Now use reference_start_time in your function calls
    closest_start_time = find_closest_time(pd.to_timedelta(start_tag), ibi_data, reference_start_time)
    closest_end_time = find_closest_time(pd.to_timedelta(end_tag), ibi_data, reference_start_time)
    
    formatted_start_time = format_time_for_display(closest_start_time, reference_start_time)
    formatted_end_time = format_time_for_display(closest_end_time, reference_start_time)

    
    st.write("Selected Start Time:", formatted_start_time)
    st.write("Selected End Time:", formatted_end_time)

    # Extract and display the length of the BVP segment
    segment = bvp_data[(bvp_data['Timestamp'] >= closest_start_time) & (bvp_data['Timestamp'] <= closest_end_time)]
    st.write("Length of BVP Segment before removing gaps:", len(segment))

    # Find and display gaps
    segment = bvp_data[(bvp_data['Timestamp'] >= closest_start_time) & (bvp_data['Timestamp'] <= closest_end_time)]
    ibi_segment = ibi_data[(ibi_data['Timestamp'] >= closest_start_time) & (ibi_data['Timestamp'] <= closest_end_time)]
    # Now find and display gaps
    gaps = find_gaps(ibi_segment['IBI'])
    st.write("Identified Gaps:", gaps)

    # Remove gaps from the BVP segment and process
    bvp_segment_without_gaps = remove_gaps_from_bvp(segment, gaps)

    # Check if the segment is long enough for processing
    if len(bvp_segment_without_gaps) > 21:  # Ensure the segment is longer than the padlen
        # Process the BVP segment and compute HRV metrics
        hrv_metrics, cleaned_bvp, r_peaks = process_and_analyze_bvp(bvp_segment_without_gaps, 64)
        st.write(hrv_metrics)
    else:
        # Handle the case where the segment is too short
        st.error("The selected BVP data segment is too short for analysis after removing gaps. Please select a longer duration or different event.")

