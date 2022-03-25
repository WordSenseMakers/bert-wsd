import click, colorama

import datagen, modelling, raw

if __name__ == "__main__":
    colorama.init(autoreset=True)
    main = click.Group(commands=[modelling.main, raw.main, datagen.main])
    main()
