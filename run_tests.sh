#!/bin/bash
# Helper script to run tests with the virtual environment

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run pytest with all arguments passed through
pytest "$@"
