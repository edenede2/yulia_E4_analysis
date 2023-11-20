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
    initial_timestamp_line = uploaded_file.readline().decode().strip()
    initial_timestamp_float = float(initial_timestamp_line.split(',')[0].strip())
    initial_timestamp = pd.to_datetime(initial_timestamp_float, unit='s', utc=True)  # Convert to datetime
    sample_rate = float(uploaded_file.readline().decode().strip().split(',')[0])
    bvp_data = pd.read_csv(uploaded_file, header=None)
    bvp_data['Timestamp'] = initial_timestamp + pd.to_timedelta(bvp_data.index / sample_rate, unit='s')
    return bvp_data, initial_timestamp, sample_rate


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


def find_gaps(ibi_data, threshold, bvp_initial_timestamp):
    gap_indices = []
    gap_details = []  # Store detailed gap information for display
    for i in range(1, len(ibi_data)):
        time_diff = (ibi_data.iloc[i]['Timestamp'] - ibi_data.iloc[i - 1]['Timestamp']).total_seconds()
        if time_diff > threshold:
            gap_indices.append(i)
            gap_start = ibi_data.iloc[i - 1]['Timestamp']
            gap_end = ibi_data.iloc[i]['Timestamp']
            gap_duration = time_diff
            gap_details.append({
                'start': format_time_for_display(gap_start, bvp_initial_timestamp),
                'end': format_time_for_display(gap_end, bvp_initial_timestamp),
                'duration': str(datetime.timedelta(seconds=int(gap_duration)))
            })
    return gap_indices, gap_details




    
def remove_gaps_from_bvp(bvp_data, ibi_data, gaps, bvp_sampling_rate):
    for index in gaps:
        if index > 0 and index < len(ibi_data):
            gap_start = ibi_data.iloc[index - 1]['Timestamp']
            gap_end = ibi_data.iloc[index]['Timestamp']
            gap_duration = (gap_end - gap_start).total_seconds()

            # Calculate the number of BVP data points to remove
            bvp_points_to_remove = int(gap_duration * bvp_sampling_rate)

            # Find the start and end indices in BVP data to remove
            bvp_start_index = bvp_data[bvp_data['Timestamp'] >= gap_start].index[0]
            bvp_end_index = bvp_start_index + bvp_points_to_remove

            # Remove the segment from BVP data
            bvp_data = bvp_data.drop(bvp_data.index[bvp_start_index:bvp_end_index])

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

def format_time_for_display(timestamp, bvp_initial_timestamp):
    elapsed_time = timestamp - bvp_initial_timestamp
    
    # Format time without the date component
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

def convert_length_to_time(length, sample_rate):
    total_seconds = length / sample_rate
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return "{:02}:{:02}:{:02}.{:03}".format(int(hours), int(minutes), int(seconds), int(milliseconds))


def analyze_hrv_from_ppg(bvp_data, ibi_data, event_start, event_end, sampling_rate, gap_threshold=4.0):
    # Convert timestamps to datetime if necessary
    bvp_data['Timestamp'] = pd.to_datetime(bvp_data['Timestamp'])
    ibi_data['Timestamp'] = pd.to_datetime(ibi_data['Timestamp'])

    # Select data segment for the event
    segment = bvp_data[(bvp_data['Timestamp'] >= event_start) & (bvp_data['Timestamp'] <= event_end)]

    # Clean the PPG signal
    cleaned_bvp = nk.ppg_clean(segment[0], sampling_rate=sampling_rate)

    # Find R-peaks in the cleaned PPG signal
    peaks_info = nk.ppg_findpeaks(cleaned_bvp, sampling_rate=sampling_rate)
    r_peaks = peaks_info['PPG_Peaks']

    # Compute HRV metrics (focusing on time-domain metrics)
    hrv_metrics = nk.hrv_time(r_peaks, sampling_rate=sampling_rate, show=False)

    # Find and handle gaps in IBI data
    gap_info = find_and_summarize_gaps(ibi_data, event_start, event_end, gap_threshold)

    return hrv_metrics, gap_info

# Helper function to find and summarize gaps
def find_and_summarize_gaps(ibi_data, start_time, end_time, threshold):
    # Filter IBI data for the selected event
    ibi_segment = ibi_data[(ibi_data['Timestamp'] >= start_time) & (ibi_data['Timestamp'] <= end_time)]

    # Find gaps
    gap_indices = []
    total_gap_duration = 0
    longest_gap = 0
    for i in range(1, len(ibi_segment)):
        time_diff = (ibi_segment.iloc[i]['Timestamp'] - ibi_segment.iloc[i - 1]['Timestamp']).total_seconds()
        if time_diff > threshold:
            gap_indices.append(i)
            total_gap_duration += time_diff
            longest_gap = max(longest_gap, time_diff)

    num_gaps = len(gap_indices)
    gap_summary = {
        "total_gaps": num_gaps,
        "total_gap_duration": total_gap_duration,
        "longest_gap": longest_gap
    }
    
    return gap_summary


# Streamlit App
st.title('HRV Metrics from PPG Data')

# File Uploaders
bvp_file = st.file_uploader("Upload BVP.csv", type="csv")
tags_file = st.file_uploader("Upload tags.csv", type="csv")
ibi_file = st.file_uploader("Upload IBI.csv", type="csv")

if bvp_file and tags_file and ibi_file:
    # Read the data and capture initial_timestamp
    bvp_data, bvp_initial_timestamp, bvp_sample_rate = read_bvp_data(bvp_file)
    tags_data, tags_initial_timestamp = read_tags_data(tags_file)
    ibi_data, ibi_initial_timestamp = read_ibi_data(ibi_file)

    # Use any of the initial timestamps (assuming they are the same)
    reference_start_time = pd.to_datetime(bvp_initial_timestamp, unit='s')
    
    # Convert the timestamp to relative time (since start of the recording)
    tags_data['Relative Time'] = tags_data['Elapsed Time'].apply(lambda x: str(datetime.timedelta(seconds=int(x))))
    event_tags = tags_data['Relative Time'].tolist()

    event_names = ["Baseline", "5-Digit test", "Exposure", "Event1", "Event2"]
    for event_name in event_names:
        st.subheader(f'Event: {event_name}')
    
        # User selects start and end tags for the chosen event
        start_tag = st.selectbox(f'Select Start Tag for {event_name}', event_tags, key=f'{event_name}_start_tag')
        end_tag = st.selectbox(f'Select End Tag for {event_name}', event_tags, key=f'{event_name}_end_tag')
        
        start_tag_timedelta = pd.to_timedelta(start_tag)
        end_tag_timedelta = pd.to_timedelta(end_tag)
        
        closest_start_time = find_closest_time(start_tag_timedelta, ibi_data, reference_start_time)
        closest_end_time = find_closest_time(end_tag_timedelta, ibi_data, reference_start_time)
        
        formatted_start_time = format_time_for_display(closest_start_time, bvp_initial_timestamp)
        formatted_end_time = format_time_for_display(closest_end_time, bvp_initial_timestamp)
    
        st.write("Selected Start Time:", formatted_start_time)
        st.write("Selected End Time:", formatted_end_time)
    
        segment = bvp_data[(bvp_data['Timestamp'] >= closest_start_time) & (bvp_data['Timestamp'] <= closest_end_time)]

        # After calculating the segment length
        length_before = len(segment)
        formatted_length_before = convert_length_to_time(length_before, bvp_sample_rate)

        ibi_segment = ibi_data[(ibi_data['Timestamp'] >= closest_start_time) & (ibi_data['Timestamp'] <= closest_end_time)]
        gap_indices, gap_info = find_gaps(ibi_segment, 4.0, bvp_initial_timestamp)
        
        for gap in gap_info:
            st.write(f"Gap from {gap['start']} to {gap['end']}, Duration: {gap['duration']}")


        # Correct the call to remove_gaps_from_bvp function
        bvp_segment_without_gaps = remove_gaps_from_bvp(segment, ibi_segment, gap_indices, bvp_sample_rate)
        length_after = len(bvp_segment_without_gaps)
        formatted_length_after = convert_length_to_time(length_after, bvp_sample_rate)

        st.write("Length of BVP Segment before removing gaps:", formatted_length_before)
        st.write("Length of BVP Segment after removing gaps:", formatted_length_after)        
        
        if len(bvp_segment_without_gaps) > 21:
            hrv_metrics, gap_info = analyze_hrv_from_ppg(bvp_data, ibi_data, closest_start_time, closest_end_time, sampling_rate=64)
            st.write(hrv_metrics)
            st.write("Total number of gaps:", gap_info['total_gaps'])
            st.write("Total duration of gaps (seconds):", gap_info['total_gap_duration'])
            st.write("Longest gap (seconds):", gap_info['longest_gap'])
        else:
            st.error("The selected BVP data segment is too short for analysis after removing gaps. Please select a longer duration or different event.")
