import colorama

def _colourize(colour: colorama.ansi.AnsiCodes, msg: str) -> str:
    print(f"{colour}{msg}")

def info(msg: str):
    _colourize(colorama.Fore.BLUE, msg)

def success(msg: str):
    _colourize(colorama.Fore.GREEN, msg)

def warning(msg: str):
    _colourize(colorama.Fore.YELLOW, msg)

def error(msg: str):
    _colourize(colorama.Fore.RED, msg)    

def neutral(msg: str):
    _colourize(colorama.Fore.WHITE, msg)