import click, colorama
import nltk

import datagen, modelling

if __name__ == "__main__":
    colorama.init(autoreset=True)
    main = click.Group(commands=[datagen.main, modelling.main])
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    main()
