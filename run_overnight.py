import subprocess
import pty
import os

def run_script(command, output_file):
    # Open the output file
    with open(output_file, "a") as file:
        # Start the command and redirect output
        master, slave = pty.openpty()
        process = subprocess.Popen(command, shell=True, stdout=slave, stderr=subprocess.STDOUT, text=True, bufsize=1, close_fds=True)
        os.close(slave)
        
        # Loop to print and write output to file simultaneously
        while True:
            try:
                line = os.read(master, 1024).decode()
                if not line:
                    break
                print(line, end='')  # Print to console
                file.write(line)  # Write to file
                file.flush()  # Ensure the line is written to the file immediately
            except OSError:
                break

        process.wait()  # Wait for the process to exit
        os.close(master)

if __name__ == "__main__":
    command1 = "PYTHONPATH=/Users/lalmeida/Documents/Purdue/HAR/src python3 scripts/centralized_multimodal_decision.py"
    run_script(command1, "multimodal.txt")
    run_script(command1, "multimodal.txt")
    run_script(command1, "multimodal.txt")
    run_script(command1, "multimodal.txt")
    run_script(command1, "multimodal.txt")

    run_script(command1, "multimodal.txt")
    run_script(command1, "multimodal.txt")
    run_script(command1, "multimodal.txt")
    run_script(command1, "multimodal.txt")
    run_script(command1, "multimodal.txt")