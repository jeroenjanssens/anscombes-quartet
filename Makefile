.PHONY: all clean install lab

all: README.md requirements.txt

README.md: anscombes-quartet.ipynb
	poetry run quarto render $< --to gfm --output $@

requirements.txt: pyproject.toml
	poetry export --without-hashes --output $@

clean:
	rm README.md requirements.txt
	rm -rf anscombes-quartet_files

install:
	poetry install

lab:
	poetry run jupyter lab