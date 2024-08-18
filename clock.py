import cv2
import numpy as np
import math
import random
import os
import csv

def draw_hand(image, center, angle, length, color, thickness):
    angle_rad = math.radians(angle)
    end_x = int(center[0] + length * math.sin(angle_rad))
    end_y = int(center[1] - length * math.cos(angle_rad))
    cv2.line(image, center, (end_x, end_y), color, thickness)

def draw_clock(hour, minute):
    width, height = 256,256
    clock_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = (width // 2, height // 2)
    radius = min(center) - 20

    hour_angle = (hour % 12 + minute / 60) * 30
    minute_angle = minute * 6

    hour_length = radius * 0.5
    minute_length = radius * 0.7
    
    draw_hand(clock_image, center, hour_angle, hour_length, (0, 0, 0), 12)
    draw_hand(clock_image, center, minute_angle, minute_length, (0, 0, 0), 10)

    return clock_image

if __name__ == '__main__':
    if not os.path.exists('images_tf'):
        os.makedirs('images_tf')
    
    number_of_images = int(input("Enter Number of Clock Images to generate : "))
    
    with open('labels.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Filename', 'hour', 'minute'])
        
        for i in range(1, number_of_images + 1):
            hour = random.randint(0, 11)
            minute = random.randint(0, 59)
            clock_image = draw_clock(hour, minute)
            image_filename = f'clock_{i}.png'
            image_path = os.path.join('images_tf', image_filename)
            
            # Save the image
            if cv2.imwrite(image_path, clock_image):
                print(f"Successfully saved {image_filename}")
            else:
                print(f"Error saving {image_filename}")

            # Write label
            writer.writerow([image_filename, hour, minute])
    
    print(f"Successfully generated {number_of_images} images and saved labels to labels.csv")
