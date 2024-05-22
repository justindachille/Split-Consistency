import subprocess
import argparse
import threading
import re
import signal
from queue import Queue

def process_commands(command_queue, device_queues, device_list):
    while True:
        command = command_queue.get()
        device = None
        while device is None:
            for d in device_list:
                if device_queues[d].empty():
                    device = d
                    break
            if device is None:
                command_queue.put(command)
                command_queue.task_done()
                break
        if device is not None:
            command = re.sub(r'--device\s+cuda:\d+', f'--device {device}', command)
            print(f"Running command on {device}: {command}")
            device_queues[device].put(command)
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed on {device}: {e}")
            finally:
                device_queues[device].get()
                command_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description='Subprocess system for running Python commands on CUDA devices.')
    parser.add_argument('--devices', type=int, nargs='+', help='List of CUDA device numbers')
    args = parser.parse_args()

    device_list = [f"cuda:{i}" for i in args.devices]
    device_queues = {device: Queue(maxsize=1) for device in device_list}
    command_queue = Queue()

    num_threads = len(device_list)
    threads = []

    for _ in range(num_threads):
        thread = threading.Thread(target=process_commands, args=(command_queue, device_queues, device_list))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    def signal_handler(sig, frame):
        print("Keyboard interrupt received. Exiting...")
        command_queue.join()
        for thread in threads:
            thread.join()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("Subprocess system started. Enter commands (one per line) or press Ctrl+C to exit.")
    print("Available commands:")
    print(" q - Print current queue sizes")
    print(" <command> - Add a command to the queue")

    while True:
        commands = input("> ")
        if commands.strip() == "q":
            print("Current command queue size:", command_queue.qsize())
            print("Current device queue sizes:")
            for device, queue in device_queues.items():
                print(f"{device}: {queue.qsize()}")
        else:
            for command in commands.split('\n'):
                command = command.strip()
                if command:
                    command_queue.put(command)

if __name__ == '__main__':
    main()