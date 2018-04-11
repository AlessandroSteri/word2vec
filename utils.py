from time import time

# class Chronometer():
class LogTime():

    def __init__(self, log_string, mode="s"):
        self.start = None
        self.log_string = log_string
        self.mode = mode
        self.unit = 1
        if mode == 's':
            self.unit = 60**0
        elif mode == 'm':
            self.unit = 60**1
        elif mode == 'h':
            self.unit = 60**2
        else:
            self.mode = 's'

    def __enter__(self):
        self.start = time()

    def __exit__(self, *args):
        print("[LogTime] {}: {}{}".format(self.log_string, (time() - self.start)/self.unit, self.mode))

def main():
    with LogTime('Dumb function elpsed time'):
        for i in range(1000000):
            pass

if __name__ == '__main__':
    main()
