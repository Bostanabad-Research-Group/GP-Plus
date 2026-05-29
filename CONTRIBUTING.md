# GPPlus Contribution Guide

## 0. Introduction

We welcome contributions from the community! This guide outlines how you can submit a pull request (PR) to contribute to the GPPlus project.

### Communicate with us

We are happy to talk about your ideas for contributing to GPPlus. Please, for any new contribution, create an issue to discuss your thoughts through [issue](https://github.com/Bostanabad-Research-Group/GP-Plus/issues) section.

## 1. Contribute to GPPlus

### Pull requests (PR)

Developer workflow for code contribution is as follows:

1. Fork the repository

- Create a fork of GPPlus repository to have your own copy of the codebase that you can modify.

2. Create a new branch

- Clone your forked repository to your local machine
- Create a new branch for your specific changes. It's recommended to name the branch descriptively, reflecting the nature of your contribution.

3. Create a Pull Request

- Make changes in your local branch following project's coding style guide.
- Push changes to your personal fork.
- Once the changes are ready for review, you can create a Pull Request to merge the changes from a branch of your fork into a branch of upstream.
    - Make sure you update the [CHANGELOG.md](./CHANGELOG.md).

4. Merging your Pull Request

- The PR will be accepted after adequate review and testing has been completed and the corresponding issue will be closed. Note that every PR should correspond to an open issue and should be linked on Github.


## 2. Code Quality and Formatting
We use Ruff to maintain code quality and ensure consistent formatting in this project. Ruff is a fast, Python-focused linter and formatter.

### 2.1. Liniting
To check the code for any linting issues, run the following command:

```bash
ruff check . --fix
```
This command will analyze the Python code in the current directory (.) and report any issues such as style violations or potential errors.

### 2.2. Formatting
To automatically format the code and fix any formatting issues, run:

```bash
ruff format .
```
This will reformat the Python files in the current directory to conform to standard style guidelines.

#### Installation
To use ruff, first, make sure it is installed in your environment. You can install it via pip:

```bash
pip install ruff
```

### 2.3. Running Tests and Checking Code Coverage

This guide explains how to run unit tests and integration tests and check the code coverage for the project.

#### Installing Dependencies

To install the required dependencies for running tests and checking coverage, use the following command:

```bash
pip install pytest pytest-cov
```

#### Running Tests
To run the tests, follow these steps:

- Open a terminal and navigate to the root directory of the project.

- Run the unit tests or integration tests using pytest.

```bash
pytest test/unit
```

```bash
pytest test/integration
```
This will run all the unit tests or integration tests and `pytest` will automatically discover and run any files that start with test_ or end with _test.py.

#### Checking Code Coverage
To check the code coverage while running the tests, use the --cov option with pytest. For example:

```bash
pytest --cov=gpplus test
```

For specific coverage of unit or intergation tests, use `pytest --cov=gpplus test/unit` or `pytest --cov=gpplus test/integration`

#### Generating Requirements File with pipreqs

To automatically generate a requirements.txt file containing only the dependencies used in your project, follow these steps:

##### Installation

Ensure you have pipreqs installed. If not, install it using:

```bash
pip install pipreqs
```

##### Usage

Run the following command in your project's root directory:

```bash
pipreqs . --force
```

The . specifies the current directory.

The --force flag overwrites any existing requirements.txt file.

## 3. Documentation

We use [MkDocs](https://www.mkdocs.org/) to geneate our documentation. To run it locally, please follow the next steps.

- Install required dependencies.

```bash
pip install mkdocs mkdocs-material mkdocs-autorefs mkdocs-jupyter mkdocs-include-markdown-plugin mkdocstrings[python]
```

- Run locally at `localhost:8000`.

```bash
mkdocs serve
```