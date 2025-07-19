#!/usr/bin/env bash
# exit on error
set -o errexit

# Use the server-specific requirements file
pip install -r requirements_server.txt
