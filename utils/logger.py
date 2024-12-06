RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
END = '\033[00m'

class Logger:
    
    def debug(self, s) :
      print(f'[{YELLOW}DEBUG{END}] {s}')

    def info(self, s) -> None:
        print(f'[{GREEN}INFO{END}] {s}')

    def error(self, s) -> None:
        print(f'[{RED}ERROR{END}] {s}')
