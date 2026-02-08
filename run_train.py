"""Wrapper to run training with output saved to a log file."""
import sys
import os

# Redirect stdout/stderr to a log file AND to console
class Tee:
    def __init__(self, filename, stream):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stream = stream
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        try:
            self.stream.write(data)
            self.stream.flush()
        except:
            pass
    def flush(self):
        self.file.flush()
        try:
            self.stream.flush()
        except:
            pass

log_path = os.path.join(os.path.dirname(__file__), 'results', 'logs', 'training_output.log')
sys.stdout = Tee(log_path, sys.stdout)
sys.stderr = Tee(log_path, sys.stderr)

# Run training
from train import train
train()
