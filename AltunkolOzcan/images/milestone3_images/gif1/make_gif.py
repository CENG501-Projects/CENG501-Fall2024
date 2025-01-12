from PIL import Image
import os

def create_high_quality_gif(input_folder, output_gif_path, duration=100, loop=0):
    """
    Create a high-quality GIF from PNG images sorted by a custom naming convention.
    
    Args:
        input_folder (str): Folder containing the PNG images.
        output_gif_path (str): Path to save the output GIF.
        duration (int): Duration for each frame in milliseconds.
        loop (int): Number of loops. 0 for infinite looping.
    """
    # Get list of PNG files in the input folder
    images = [
        os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.png')
    ]
    
    # Custom sorting based on numeric part of the filename
    def sort_key(filename):
        # Extract the numeric part after 'g1.' in the filename
        base_name = os.path.basename(filename)
        try:
            # Split and convert the part after 'g1.' to a float for proper sorting
            return float(base_name.split('g1.')[1].replace('.png', ''))
        except (IndexError, ValueError):
            return float('inf')  # Push invalid files to the end
    
    images.sort(key=sort_key)
    
    # Ensure there are images to process
    if not images:
        print("No PNG files found in the input folder.")
        return
    
    # Open images and save as GIF
    frames = [Image.open(img) for img in images]
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration,
        loop=loop
    )
    print(f"GIF saved to {output_gif_path}")

# Parameters
input_folder = ".\\"  # Replace with your folder path
output_gif_path = "output.gif"            # Replace with your desired output path
duration = 1000                            # 100ms per frame
loop = 0                                  # Infinite loop

create_high_quality_gif(input_folder, output_gif_path, duration, loop)