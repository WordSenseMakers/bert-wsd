import click

import recasting, modelling

if __name__ == "__main__":
    main = click.Group(commands=[modelling.main, recasting.main])
    main()
