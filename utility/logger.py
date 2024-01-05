



class Logger:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.log_file = open(self.path + self.name, 'w')

    def log(self, message):
        self.log_file.write(message + '\n')

    def close(self):
        self.log_file.close()