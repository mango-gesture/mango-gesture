# A simple script to read bytes from a file and store a jpeg file.
# By Nika Zahedi

import numpy as np

data_dir = 'data/swipe_left'

with open(f'bg.jpg', 'rb') as file:
    bg_img = file.read()

def find_bit_sequence(data, byte1, byte2):
    for i in range (len(data) - 1):
        if data[i] == byte1 and data[i + 1] == byte2:
            return i

    raise ValueError(f"JPEG marker not found in the data.")


def find_jpeg_markers(data):
    start_index = find_bit_sequence(data, 0xFF, 0xD8)
    end_index = find_bit_sequence(data, 0xFF, 0xD9)

    return start_index, end_index + 2

# Read the captured file data
with open('capture.txt', 'rt', encoding='utf-8', errors='ignore') as file:
    data_str = file.read()
    images_as_str = data_str.split('Size ')[1:] # The first element is not an image, so we ignore it
    images = [[ord(im_str[i]) - ord('a') + (ord(im_str[i + 1]) - ord('a')) * 26 for i in range (0, len(im_str) - 1, 2)] for im_str in images_as_str]

# Find the JPEG markers
try:
    max_size = -1

    first_images_in_pair = images[::2]
    second_images_in_pair = images[1::2]

    num = 0
    for i in range (len(first_images_in_pair)):
        start_index_second, end_index_second = find_jpeg_markers(second_images_in_pair[i])
        if (end_index_second - start_index_second + 1 - len(bg_img)) < 90:
            continue
        jpeg_data_second = bytes(second_images_in_pair[i][start_index_second:end_index_second])

        start_index_first, end_index_first = find_jpeg_markers(first_images_in_pair[i])
        jpeg_data_first = bytes(first_images_in_pair[i][start_index_first:end_index_first])

        max_size = max(max_size, max(len(jpeg_data_first), len(jpeg_data_second)))

        # Save the extracted JPEG data to a new file
        with open(f'{data_dir}/output{2 * num}.jpg', 'wb') as jpeg_file:
            jpeg_file.write(jpeg_data_first)
        with open(f'{data_dir}/output{2 * num + 1}.jpg', 'wb') as jpeg_file:
            jpeg_file.write(jpeg_data_second)
        num += 1
    # for num, data in enumerate(images):
    #     start_index, end_index = find_jpeg_markers(data)

    #     if (end_index - start_index + 1) < 100:
    #         continue
    #     jpeg_data = bytes(data[start_index:end_index])

    #     max_size = max(max_size, len(jpeg_data))
    #     # Save the extracted JPEG data to a new file
    #     with open(f'{data_dir}/output{num}.jpg', 'wb') as jpeg_file:
    #         jpeg_file.write(jpeg_data)
    #     print(f"JPEG file extracted and saved as output{num}.jpg")
        
    with open(f'{data_dir}/log.txt', 'wt') as data_log:
        data_log.write(f"Max size: {max_size}")

except ValueError as e:
    print(e)

