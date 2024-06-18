# Main_AI
# Project Name

This project utilizes various libraries for natural language processing and machine learning. Below are the steps to set up the environment and run the project.

## Prerequisites

Make sure you have the following installed:
- Python 3.9 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/yourproject.git
    cd yourproject
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your data: Make sure your data is in the correct format and located in the appropriate directory.

2. Run the script: Execute your main Python script to start processing the data.

    ```bash
    python test_model.py
    ```

## Libraries Used

- **os**: Provides functions to interact with the operating system.
- **random**: Implements pseudo-random number generators.
- **scikit-learn**: A machine learning library for Python, used for data splitting and more.
- **transformers**: Provides general-purpose architectures for NLP, including tokenizers and models.
- **tensorflow**: An end-to-end open-source platform for machine learning.
- **pandas**: A library providing data structures and data analysis tools.
- **numpy**: A library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- **pdfplumber**: A library for extracting text, tables, and metadata from PDF files.
- **re**: A library for regular expression matching operations.
- **csv**: A library for reading and writing CSV files.

## Example

Below is an example of how to use the main functionalities of this project:

```python
import os
import random
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
import pdfplumber
import re
import csv


### Summary

1. **`requirements.txt`**: Lists all necessary packages and their versions.
2. **`README.md`**: Provides detailed instructions on how to set up and run the project, including installation steps and an example usage section.

Make sure to replace `yourusername` and `yourproject` with your actual GitHub username and project name, respectively. Additionally, update the example code section with relevant code from your project.
