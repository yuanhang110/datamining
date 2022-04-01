---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Introduction

+++

```{tip}
The introduction section should identify the issues that motivates the current work. 
```

+++

For research work that involves coding, traditional publications in static formats are not ideal:

- A paper does not allow readers to interact with the code to learn the results.
- Computer source code is hard to read as comments are in plain text and often very brief.
- In case of error in the code, changes need to be made separately to the paper.

+++

Jupyter Notebooks solve the above problems by combining richly formatted markdown cells with executable code cells. We can further use:

- [Jupyter Book](https://jupyterbook.org/) to compile executable contents in multiple Jupyter Notebooks with proper citations.
- [Docker image](https://hub.docker.com/) to run the notebooks anywhere with the required dependencies automatically set up.
- [GitLab](https://docs.gitlab.com/) to version control and publish the work.

+++ {"tags": []}

This work is a demonstration and template for compiling a JupyterBook on GitLab that can run iteractively using a docker image for a data science course.
