from setuptools import setup    # type: ignore

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()


AUTHOR_NAME = 'BARBOD SADRAEIFAR'
SRC_REPO = 'src'
LIST_OF_REQUIREMENTS = ['streamlit', 'numpy', 'pandas', 'langchain', 'langchain_groq']

setup(name = SRC_REPO,
    version = '0.0.1',
    author = AUTHOR_NAME,
    author_email = 'barboad1380@gmail.com',
    description = 'A simple python package to make a simple web app',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    package = [SRC_REPO],
    python_requires = ">=3.12",
    install_reuires = LIST_OF_REQUIREMENTS)