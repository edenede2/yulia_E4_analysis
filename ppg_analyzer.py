import streamlit as st
import pandas as pd
import datetime
import neurokit2 as nk

# Helper Functions
def convert_to_elapsed_time(df, initial_timestamp):
    # Convert initial timestamp to datetime
    initial_time = pd.to_datetime(initial_timestamp, unit='s')
    
    # Calculate time elapsed since initial timestamp for each row
    df['Elapsed Time'] = df.index / df.iloc[1, 0] + initial_timestamp
    df['Elapsed Time'] = (initial_time + pd.to_timedelta(df['Elapsed Time'], unit='s')).dt.strftime('%H:%M:%S:%f')
    return df

def find_closest_time(event_time, ibi_data):
    # Convert event_time to a datetime object for comparison
    event_datetime = datetime.datetime.strptime(event_time, '%H:%M:%S:%f')
    
    # Convert IBI times to datetime for comparison
    ibi_data['Time'] = pd.to_datetime(ibi_data['Time'], format='%H:%M:%S:%f')
    
    # Find the row with the closest time to the event time
    closest_time = ibi_data.iloc[(ibi_data['Time'] - event_datetime).abs().argsort()[:1]]
    return closest_time

def process_bvp_signal(bvp_data, sampling_rate):
    # Clean the PPG signal
    processed_signal = nk.ppg_clean(bvp_data, sampling_rate=sampling_rate)
    return processed_signal


# Streamlit App
st.title('HRV Metrics from PPG Data')

# File Uploaders
bvp_file = st.file_uploader("Upload BVP.csv", type="csv")
tags_file = st.file_uploader("Upload tags.csv", type="csv")
ibi_file = st.file_uploader("Upload IBI.csv", type="csv")
acc_file = st.file_uploader("Upload ACC.csv", type="csv")

if bvp_file and tags_file and ibi_file and acc_file:
    bvp_data = pd.read_csv(bvp_file, header=None)
    tags_data = pd.read_csv(tags_file, header=None)
    ibi_data = pd.read_csv(ibi_file, header=None)
    acc_data = pd.read_csv(acc_file, header=None)

    # Convert time to time from start
    initial_timestamp = datetime.datetime.now().timestamp()  # Replace with actual initial timestamp
    bvp_data = convert_to_elapsed_time(bvp_data, initial_timestamp)
    ibi_data = convert_to_elapsed_time(ibi_data, initial_timestamp)
    # Do the same for other dataframes as needed

    # User interaction for event selection
    event_name = st.selectbox('Select Event', ['Baseline', '5-Digit test', 'Exposure', 'Event1', 'Event2'])
    event_time = st.text_input('Enter Event Time (HH:MM:SS:MS)')

    if event_time:
        closest_time = find_closest_time(event_time, ibi_data)
        
    # Find corresponding time in IBI data
    closest_time = find_closest_time(event_time, ibi_data)

    # Process BVP Signal
    processed_bvp = process_bvp_signal(bvp_data, 64)  # Assuming 64Hz sampling rate

    # Calculate HRV metrics
    hrv_metrics = nk.hrv(processed_bvp, sampling_rate=64, show=True)
    st.write(hrv_metrics)

    # Visualization (if needed)
    st.line_chart(processed_bvp)

    # Download results
    st.download_button(label="Download HRV Metrics as CSV", data=hrv_metrics.to_csv(), file_name='hrv_metrics.csv', mime='text/csv')
