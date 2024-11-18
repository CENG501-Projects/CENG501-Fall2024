import numpy as np
import os
import h5py
import cv2
import torch
import matplotlib.pyplot as plt

def torch_to_cv2_image(torch_image):
    """
    Converts a PyTorch tensor image to a format compatible with OpenCV.
    """
    np_image = torch_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Change [C, H, W] to [H, W, C]
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return cv2_image


def plot_points_on_background(points_coordinates, background, points_color=[0, 0, 255]):
    """
    Plots points on a given background image.
    Args:
        points_coordinates: array of (y, x) points coordinates
                            of size (number_of_points x 2).
        background: (3 x height x width) gray or color image uint8.
        points_color: color of points [red, green, blue] uint8.
    """
    if not (len(background.size()) == 3 and background.size(0) == 3):
        raise ValueError('background should be (color x height x width).')
    _, height, width = background.size()
    background_with_points = background.clone()
    y, x = points_coordinates.transpose(0, 1)
    if len(x) > 0 and len(y) > 0:  # There can be empty arrays!
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
            raise ValueError('points coordinates are outside the "background" boundaries.')
        background_with_points[:, y, x] = torch.Tensor(points_color).type_as(background).unsqueeze(-1)
    return background_with_points


def events_to_event_image(event_sequence):
    """
    Converts an event sequence to an event image visualization with colors.
    """
    width = 346
    height = 260
    polarity = event_sequence[:, 3] == -1.0
    x_negative = event_sequence[~polarity, 0].astype(int)
    y_negative = event_sequence[~polarity, 1].astype(int)
    x_positive = event_sequence[polarity, 0].astype(int)
    y_positive = event_sequence[polarity, 1].astype(int)

    positive_histogram, _, _ = np.histogram2d(
        x_positive, y_positive, bins=(width, height), range=[[0, width], [0, height]])
    negative_histogram, _, _ = np.histogram2d(
        x_negative, y_negative, bins=(width, height), range=[[0, width], [0, height]])

    # Red -> Negative Events
    red = np.transpose((negative_histogram >= positive_histogram) & (negative_histogram != 0))
    # Blue -> Positive Events
    blue = np.transpose(positive_histogram > negative_histogram)

    # Create a background image
    background = torch.full((3, height, width), 255).byte()

    # Visualize the events with colored points
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(red.astype(np.uint8))), background, [255, 0, 0])
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(blue.astype(np.uint8))), points_on_background, [0, 0, 255])

    # Convert image into OpenCV format
    points_on_background = torch_to_cv2_image(points_on_background)

    return points_on_background


def visualize_velodyne_data(velodyne_data):
    """
    Visualizes Velodyne point cloud data in a 3D plot.
    Args:
        velodyne_data: Velodyne point cloud (N, 4), where N is the number of points.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z, intensity (assuming the format is x, y, z, intensity)
    x = velodyne_data[:, 1]
    y = velodyne_data[:, 2]
    z = velodyne_data[:, 0]
    intensity = velodyne_data[:, 3]

    ax.scatter(x, y, z, c=intensity, cmap='jet', marker='o', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Velodyne 3D Point Cloud")
    #plt.show()
    
    # Save the 3D plot to an image file (temporary)
    plt.savefig('temp_velodyne_plot.png')
    plt.close(fig)  # Close the figure after saving to prevent memory issues
    
    # Load the saved plot as an image
    velodyne_image = cv2.imread('temp_velodyne_plot.png')
    
    return velodyne_image


def stack_images(grayscale, event, velodyne):
    """
    Stacks the grayscale, event image, and Velodyne image horizontally and vertically.
    """
    # Resize the images to ensure they all have the same size
    event_resized = cv2.resize(event, (grayscale.shape[1], grayscale.shape[0]))
    velodyne_resized = cv2.resize(velodyne, (grayscale.shape[1], grayscale.shape[0]))

    # Stack the images vertically
    top_row = np.hstack((grayscale, event_resized))
    bottom_row = np.hstack((velodyne_resized, np.zeros_like(grayscale)))  # Add a placeholder if needed

    # Stack vertically
    stacked_images = np.vstack((top_row, bottom_row))
    return stacked_images


if __name__ == "__main__":
    # Path to data
    path_to_data = os.path.join("", "indoor_flying1_data.hdf5")

    # Load the data
    hdf_file = h5py.File(path_to_data, "r")

    # Check the structure of the data
    print("Top-level groups:", list(hdf_file.keys()))

    # Load grayscale data
    gray_image_data = hdf_file['davis']['left']['image_raw']
    print("Grayscale shape:", gray_image_data.shape)

    # Load event data
    events_data = hdf_file['davis']['left']['events']
    print("Events shape:", events_data.shape)

    # Load Velodyne scans
    velodyne_scans = hdf_file['velodyne']['scans']
    print("Velodyne scans shape:", velodyne_scans.shape)

    # Load event to image indexes
    image_raw_event_inds = hdf_file['davis']['left']['image_raw_event_inds']

    # Process and display frames live
    for idx in range(1000, 2000):  # Adjust range as needed
        # Get the grayscale image
        img_frame = gray_image_data[idx]

        # Apply histogram equalization
        img_frame = cv2.equalizeHist(img_frame)

        # Convert to BGR format for OpenCV visualization
        img_frame_bgr = cv2.cvtColor(img_frame, cv2.COLOR_GRAY2BGR)

        # Get the related events
        related_events = events_data[image_raw_event_inds[idx - 1]:image_raw_event_inds[idx + 1], :]

        # Visualize the events in color
        event_image = events_to_event_image(related_events)

        # Visualize the Velodyne data
        velodyne_data = velodyne_scans[idx, :, :]  # Assuming each scan is in shape (N, 4)
        velodyne_image = visualize_velodyne_data(velodyne_data)

        # Stack the images into one partitioned window
        stacked_images = stack_images(img_frame_bgr, event_image, velodyne_image)

        # Display the partitioned window
        cv2.imshow("Partitioned View", stacked_images)
        cv2.imshow("Partitioned View2", event_image)

        # Wait for a key press to proceed to the next frame
        key = cv2.waitKey(1)  # Adjust the delay between frames (100ms here)
        if key == 27:  # Press 'Esc' to exit
            break

    # Release resources
    cv2.destroyAllWindows()

    print("Live video display complete.")
