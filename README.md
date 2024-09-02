# Intelligent Candidate Matching System

## Project features

- RAG
- ReAct Agent
- Subquestion Query Engine
- Agentic Splitting
- Caching

## Prerequisites

- [python 3.11.4](https://www.python.org/downloads/release/python-3114/)

## Setup

To setup the project quickly, simply paste the commands below. This assumes you have [aws-cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) setup. Otherwise, you can follow a step-by-step tutorial below

```sh
git clone https://github.com/UA-AICore/joyfuljobs.git && cd joyfuljobs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env_default .env
```

Then edit your `.env` file with the appropriate values for each key.

### Cloning the repository

```sh
git clone https://github.com/UA-AICore/joyfuljobs.git
```

### Installation

Open the terminal in the root directory of this repository. Create a python virtual environment called `.venv` so you do not mess up existing python dependencies installed in your system:

```sh
python -m venv .venv
```

To virtualize your python environment so that your terminal has a `(.venv)` prefix, run:

```sh
source .venv/bin/activate
```

Then install the required dependencies via:

```sh
pip install -r requirements.txt
```

### Setting up environmental variables

Copy `.env_default` to another file called `.env` within the root directory. Put the appropriate values for each key in the file.
