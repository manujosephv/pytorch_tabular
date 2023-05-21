#!/bin/bash

# Prompt the user for their name.
echo "What is the name of the environment?"
read -r name

mkdir -p .env
# Create a virtual environment
python3 -m venv .env/$name

# Activate the virtual environment
source .env/$name/bin/activate

# Create a temporary requirements file
# Read the contents of the file into a variable.
contents=$(cat requirements.txt)
# Replace all occurrences of ">=" with "==" in the variable.
contents=$(echo "$contents" | sed 's/>=$/==/g')
# Write the contents of the variable to the file.
echo "$contents" > requirements.tmp

# Install the required dependencies from the temporary file
pip install -r requirements.tmp

rm requirements.tmp

# Install an editable version of the package
pip install -e .[dev]
