from media_ctrl import run_media_ctrl
from argparse import ArgumentParser
import os
import time

def watch_log_file(args):
    file_name = args.file
    # Get the initial size of the file
    file_size = os.path.getsize(file_name)

    while True:
        try:
            # Check if the file size has changed
            if os.path.getsize(file_name) > file_size:
                with open(file_name, 'r', encoding='utf-8', errors='ignore') as file:
                    # read the last line
                    lines = file.readlines()
                    try:
                        last_line = lines[-2]
                    except IndexError:
                        continue
                    # check if last line ends with Choice: 0 or Choice: 1
                    if last_line.endswith('Choice: 0\n') or last_line.endswith('Choice: 1\n'):
                        # get the choice, which is the last character of the stripped line
                        choice = int(last_line.strip()[-1])
                        print(f'Choice: {choice}')
                        # run the media control function
                        run_media_ctrl(choice)

                # Update the file size
                file_size = os.path.getsize(file_name)

            # wait for 10 ms
            time.sleep(0.01)
        except KeyboardInterrupt:
            os.remove(file_name)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    print(f"Watching log file: {args.file}")

    watch_log_file(args)