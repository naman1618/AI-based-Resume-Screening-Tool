# Joyful Jobs

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

### AWS

Setup [aws-cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) with the proper aws access key and secret. Check if you have them by running

```sh
cat ~/.aws/credentials
```

You should see an output that matches the format:

```
[default]
aws_access_key_id = ACCESS_KEY_ID
aws_secret_access_key = MY_SECRET_ACCESS_KEY
```

If you do not have any credentials stored, run `aws configure` in your terminal and store the appropriate variables.

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

### Running the program

There is no production-ready application, yet. The prototype application uses [gradio](https://www.gradio.app/), and before running it, **please make sure you have a `.cache` folder containing all the processed information of resumes (ask Joseph for the download link)**. Each document goes through in-depth analysis and gets parsed in a way for an LLM to understand, and this task is NOT cheap. The contents of the `.cache` file should look like this:

```
.cache/
├── 0c26b4af755bc4d0a1ef431d9adaf9a675f01c4b4decd98e8811d26e7c0b4f93-chunks.json
├── 0c26b4af755bc4d0a1ef431d9adaf9a675f01c4b4decd98e8811d26e7c0b4f93.json
├── 0c26b4af755bc4d0a1ef431d9adaf9a675f01c4b4decd98e8811d26e7c0b4f93.md
└── ...
```

The `.cache` folder should be located in the root directory of this repository (do not put it into `src`)

Once you have the `.cache` folder, you can run the program via:

```py
python src/experimental.py
```

This will start a gradio server, allowing you to interact with the chat bot.
