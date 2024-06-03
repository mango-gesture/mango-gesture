from inference.media_ctrl import run_media_ctrl
from argparse import ArgumentParser
import os
import time

def watch_log_file(args):
    file_name = args.file
    # Get the initial size of the file
    file_size = os.path.getsize(file_name)

    while True:
        # Check if the file size has changed
        if os.path.getsize(file_name) > file_size:
            with open(file_name, 'r') as file:
                # read the last 

            # Update the file size
            file_size = os.path.getsize(file_name)

        # wait for 10 ms
        time.sleep(0.01)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    watch_log_file(args)