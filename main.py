import click, colorama

import datagen, modelling

if __name__ == "__main__":
    colorama.init(autoreset=True)
    main = click.Group(commands=[modelling.main, datagen.main])
    main()
