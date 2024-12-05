import numpy as np
import cv2
import pandas as pd
import os

def load_events(csv_file):
    # Load the CSV file containing the events
    events = pd.read_csv(csv_file)
    return events

def create_event_image(events, frame_height, frame_width):
    # Create an empty frame to visualize the events
    event_image = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Set the pixel intensity based on the event polarity
    for idx, event in events.iterrows():
        x = int(event['x'])
        y = int(event['y'])
        if 0 <= x < frame_width and 0 <= y < frame_height:
            if event['polarity'] == True:
                event_image[y, x] = 255  # white for polarity True
            else:
                event_image[y, x] = 0  # black for polarity False

    return event_image

def create_video_from_events(events, frame_height, frame_width, video_file, time_window=0.03):
    # Create a VideoWriter to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec
    fps = 30  # Frames per second
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))

    # Process events in batches according to the time window
    start_time = events.iloc[0]['secs'] + events.iloc[0]['nsecs'] / 1e9
    batch_events = []
    
    for idx, event in events.iterrows():
        event_time = event['secs'] + event['nsecs'] / 1e9
        if event_time - start_time <= time_window:
            batch_events.append(event)
        else:
            # Convert events to image and write to video
            event_image = create_event_image(pd.DataFrame(batch_events), frame_height, frame_width)
            video_writer.write(event_image)
            # Reset batch for next time window
            start_time = event_time
            batch_events = [event]

    # Write the last batch
    if batch_events:
        event_image = create_event_image(pd.DataFrame(batch_events), frame_height, frame_width)
        video_writer.write(event_image)

    # Release the video writer
    video_writer.release()

def main():
    csv_file = '/media/romer/Expansion/Seagate/MVSEC/parsed_events.csv'  # Your parsed CSV file
    video_file = '/media/romer/Expansion/Seagate/MVSEC/indoor_flying_video.avi'  # Output video file
    
    frame_height = 240  # Set the height of your frame (adjust as needed)
    frame_width = 304  # Set the width of your frame (adjust as needed)

    # Load the events
    events = load_events(csv_file)
    
    # Create video from events
    print("Creating video from events...")
    create_video_from_events(events, frame_height, frame_width, video_file)
    print(f"Video saved to {video_file}")

if __name__ == '__main__':
    main()
