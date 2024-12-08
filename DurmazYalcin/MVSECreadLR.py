import os
import rosbag
from tqdm import tqdm

def read_events_from_bag(bag_file, topic_name):
    """Reads events from a ROS bag file."""
    events = []
    print(f"[INFO] Reading events from bag file: {bag_file}")
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            # Get total number of messages for progress tracking
            total_messages = bag.get_message_count(topic_filters=[topic_name])
            for topic, msg, t in tqdm(bag.read_messages(topics=[topic_name]), total=total_messages):
                for event in msg.events:
                    try:
                        event_data = {
                            'x': event.x,
                            'y': event.y,
                            'polarity': event.polarity,
                            'secs': event.ts.secs,
                            'nsecs': event.ts.nsecs
                        }
                        events.append(event_data)
                    except AttributeError as e:
                        print(f"[WARNING] Missing keys in event: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to read bag file: {e}")
    print(f"[INFO] Finished parsing events. Total events: {len(events)}")
    return events

def save_events_to_csv(events, output_file):
    """Saves parsed events to a CSV file."""
    import csv

    if not events:
        print("[ERROR] No events to save.")
        return
    
    keys = ['x', 'y', 'polarity', 'secs', 'nsecs']
    print(f"[INFO] Saving events to {output_file}")
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(events)
        print("[INFO] Events saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save events to CSV: {e}")

def main():
    """Main function to parse events from a bag file and save them."""
    # Paths and parameters
    base_path = "/media/romer/Expansion/Seagate/MVSEC"
    bag_file = os.path.join(base_path, "indoor_flying1_data.bag")
    output_csv = os.path.join(base_path, "parsed_events.csv")
    topic_name = "/davis/left/events"

    # Verify bag file existence
    if not os.path.exists(bag_file):
        print(f"[ERROR] Bag file not found: {bag_file}")
        return

    # Read events from the bag file
    events = read_events_from_bag(bag_file, topic_name)
    if not events:
        print("[ERROR] No events were parsed. Please verify the bag file and topic name.")
        return

    # Save parsed events to CSV
    save_events_to_csv(events, output_csv)

if __name__ == "__main__":
    main()
