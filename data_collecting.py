import argparse
import os
import time
import socket
from muselsl import stream, record
from muselsl.stream import list_muses

def send_udp_signal(ip, port, message):
    """
    Send a UDP signal to the specified IP address and port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode(), (ip, port))
    sock.close()

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='EEG data recording script')
    parser.add_argument('filename', type=str, help='Output CSV file name')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address to send UDP signal to')
    parser.add_argument('--port', type=int, default=5005, help='Port to send UDP signal to')
    args = parser.parse_args()

    # Set the directory to the current working directory
    directory = os.getcwd()
    filepath = os.path.join(directory, args.filename)

    # Check if file already exists
    if os.path.exists(filepath):
        print(f"Warning: The file '{args.filename}' already exists.")
        overwrite = input("Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled.")
            return

    # Check if any Muses are available
    muses = list_muses()
    if not muses:
        print('No Muses found')
        return

    # Stream from the first available Muse
    address = muses[0]['address']
    print(f'Starting stream from Muse: {muses[0]["name"]} at address {address}')
    
    # Stream data
    # stream(address=address)
    time.sleep(2)  # Give the stream a moment to start

    # Send UDP signal to Unity project
    udp_message = 'START'
    send_udp_signal(args.ip, args.port, udp_message)
    print(f"Sent UDP signal to {args.ip}:{args.port} with message '{udp_message}'")

    # Record data
    print(f'Recording EEG data to {filepath}')
    record(duration=60,  # Record for 60 seconds
           filename=filepath, 
           dejitter=True)
    
    print('Recording complete.')

if __name__ == '__main__':
    main()
