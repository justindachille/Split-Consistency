import subprocess
import argparse
import threading
import re
import signal
from queue import Queue

def process_commands(device_queues):
    while True:
        for device, queue in device_queues.items():
            if not queue.empty():
                command = queue.get()
                print(f"Running command on {device}: {command}")
                subprocess.run(command, shell=True)
                queue.task_done()

def main():
    parser = argparse.ArgumentParser(description='Subprocess system for running Python commands on CUDA devices.')
    parser.add_argument('--num_devices', type=int, default=8, help='Number of CUDA devices')
    args = parser.parse_args()

    device_queues = {f"cuda:{i}": Queue() for i in range(args.num_devices)}
    device_pattern = re.compile(r'--device\s+cuda:(\d+)')

    num_threads = len(device_queues)
    threads = []

    for _ in range(num_threads):
        thread = threading.Thread(target=process_commands, args=(device_queues,))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    def signal_handler(sig, frame):
        print("Keyboard interrupt received. Exiting...")
        for queue in device_queues.values():
            queue.join()
        for thread in threads:
            thread.join()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("Subprocess system started. Enter commands (one per line) or press Ctrl+C to exit.")
    print("Available commands:")
    print("  q - Print current queue sizes")
    print("  <command> - Add a command to the queue")

    while True:
        command = input("> ")
        if command.strip() == "q":
            print("Current queue sizes:")
            for device, queue in device_queues.items():
                print(f"{device}: {queue.qsize()}")
        else:
            match = device_pattern.search(command)
            if match:
                device = f"cuda:{match.group(1)}"
                if device in device_queues:
                    device_queues[device].put(command)
                else:
                    print(f"Invalid CUDA device specified: {device}")
            else:
                print(f"No CUDA device specified for command: {command}")

if __name__ == '__main__':
    main()