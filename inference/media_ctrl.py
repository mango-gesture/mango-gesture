import subprocess

"""
Define some media ctrl helper functions that will be targets of the gesture commands
"""

def no_op():
    pass

def skip_song():
    subprocess.run(['osascript', '-e', 'tell application "Spotify" to next track'])

def last_song():
    subprocess.run(['osascript', '-e', 'tell application "Spotify" to previous track'])

# label_to_func = {
#     0: no_op,
#     1: last_song,
#     2: skip_song,
# }

label_to_func = {
    0: last_song,
    1: skip_song,
}

def run_media_ctrl(label):
    label_to_func[label]()
    if not label:
        label_to_func[0]()
