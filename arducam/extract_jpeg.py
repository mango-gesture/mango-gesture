# A simple script to read bytes from a file and store a jpeg file.
# By Nika Zahedi

import numpy as np


def find_bit_sequence(data, byte1, byte2):
    for i in range (len(data) - 1):
        if data[i] == byte1 and data[i + 1] == byte2:
            return i

    raise ValueError(f"JPEG marker not found in the data.")


def find_jpeg_markers(data):
    # start_marker = [0xFF, 0xD8]
    # end_marker = [0xFF, 0xD9]

    start_index = find_bit_sequence(data, 0xFF, 0xD8)
    end_index = find_bit_sequence(data, 0xFF, 0xD9)

    return start_index, end_index + 2

# Read the captured file data
with open('capture.txt', 'rt') as file:
    data_str = file.read()
    data = []
    for i in range (0, len(data_str) - 1, 2):
        data.append(ord(data_str[i]) - ord('a') + (ord(data_str[i + 1]) - ord('a')) * 26)

# Find the JPEG markers
try:
    num = 0
    while (1):
        start_index, end_index = find_jpeg_markers(data)
        jpeg_data = bytes(data[start_index:end_index])

        # Save the extracted JPEG data to a new file
        with open(f'output{num}.jpg', 'wb') as jpeg_file:
            jpeg_file.write(jpeg_data)
            
        data = data[end_index:]
        num += 1

        print("JPEG file extracted and saved as output.jpg")
except ValueError as e:
    print(e)

