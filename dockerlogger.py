import csv
import subprocess
import time
import sys

output_file = sys.argv[1] if len(sys.argv) > 1 else "docker_stats.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Timestamp", "Container", "CPU%", "Memory%",
        "Memory Usage", "Net I/O", "Block I/O"
    ])
    while True:
        timestamp = int(time.time())
        output = subprocess.check_output([
            "docker", "stats", "--no-stream", "--format",
            "{{.Name}},{{.CPUPerc}},{{.MemPerc}},{{.MemUsage}},{{.NetIO}},{{.BlockIO}}"
        ]).decode()
        for line in output.strip().split("\n"):
            row = [timestamp] + line.split(",")
            writer.writerow(row)
        f.flush()
        time.sleep(1)