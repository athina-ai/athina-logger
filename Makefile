# display help for each command
help:
	@echo "Available commands:"
	@echo "  setup  - installs python@3.9 using brew"
	@echo "  venv   - creates a virtual environment in ./venv"
	@echo "  build  - activates venv, installs poetry, and sets up dependencies"

# setup the development environment by installing python
setup:
	brew install python@3.9

# create a virtual environment
venv:
	python3.9 -m venv venv

# activate venv, install poetry, and set up dependencies
build: venv
	. ./venv/bin/activate && \
	pip install poetry && \
	poetry install

# remove the virtual environment
clean:
	rm -rf venv
