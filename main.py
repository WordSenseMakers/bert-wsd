import click, colorama
import nltk

import datagen, modelling, raw

if __name__ == "__main__":
    colorama.init(autoreset=True)
    main = click.Group(commands=[modelling.main, raw.main, datagen.main])
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    main()
