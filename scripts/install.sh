#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print("".join(map(str, sys.version_info[:2])))')
if [ "$PYTHON_VERSION" -lt 39 ] || [ "$PYTHON_VERSION" -gt 312 ]; then
    echo "Error: Python 3.9 to 3.12 is required. Your version, $PYTHON_VERSION, is not supported." >&2
    echo "Please use a Python version between 3.9 and 3.12 (3.11 is recommended)." >&2
    exit 1
fi

if [ "$PYTHON_VERSION" -ne 312 ]; then
    echo "Warning: Python 3.12 is recommended for optimal performance and compatibility."
fi

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install it to continue." >&2
    exit 1
fi

# Check if gcloud CLI is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it from https://cloud.google.com/sdk/docs/install" >&2
    exit 1
fi

# Check for build tools
if ! command -v cmake &> /dev/null; then
    echo "Warning: cmake is not installed. Some dependencies may fail to build."
    echo "On macOS, you can install it with: brew install cmake"
    echo "On Debian/Ubuntu, you can install it with: sudo apt-get install cmake"
fi

if ! command -v pkg-config &> /dev/null; then
    echo "Warning: pkg-config is not installed. Some dependencies may fail to build."
    echo "On macOS, you can install it with: brew install pkg-config"
    echo "On Debian/Ubuntu, you can install it with: sudo apt-get install pkg-config"
fi

if ! command -v nproc &> /dev/null; then
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Error: 'nproc' command not found. It is required for building some dependencies." >&2
        echo "On macOS, you can install it by running: brew install coreutils" >&2
    else
        echo "Error: 'nproc' command not found. It is required for building some dependencies." >&2
        echo "Please install 'nproc' using your system's package manager (e.g., 'sudo apt-get install coreutils' on Debian/Ubuntu)." >&2
    fi
    exit 1
fi

# Clone ComfyUI repository
if [ ! -d "ComfyUI" ]; then
    git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git
fi
cd ComfyUI

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone the custom nodes repository
cd custom_nodes
if [ ! -d "vertex-ai-comfyui-nodes" ]; then
    git clone --depth=1 https://github.com/khanhlvg/vertex-ai-comfyui-nodes.git
    cd vertex-ai-comfyui-nodes
    pip install -r requirements.txt
    cd ..
fi

# Authenticate with GCP
if gcloud auth application-default print-access-token &> /dev/null; then
    read -p "You are already authenticated with GCP. Do you want to re-authenticate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud auth application-default login
    fi
else
    read -p "Do you want to authenticate with GCP using your credentials? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        gcloud auth application-default login
    else
        echo "Authentication skipped. The Vertex AI nodes will not work without authentication."
    fi
fi

# Set Environment Variables
echo "--- "
echo "Setting up environment variables..."

# Get current values from the venv activate script if they exist
CURRENT_PROJECT=""
if [ -f "../venv/bin/activate" ]; then
    CURRENT_PROJECT=$(grep "export GOOGLE_CLOUD_PROJECT" ../venv/bin/activate | cut -d'=' -f2 | tr -d '"' )
fi

# Handle GOOGLE_CLOUD_PROJECT
if [ -n "$CURRENT_PROJECT" ]; then
    read -p "GOOGLE_CLOUD_PROJECT is currently set to '$CURRENT_PROJECT'. Overwrite? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Please enter your new Google Cloud Project ID:"
        read project_id
    else
        project_id=$CURRENT_PROJECT
    fi
else
    echo "Please enter your Google Cloud Project ID:"
    read project_id
fi

# Get current values from the venv activate script if they exist
CURRENT_LOCATION=""
if [ -f "../venv/bin/activate" ]; then
    CURRENT_LOCATION=$(grep "export GOOGLE_CLOUD_LOCATION" ../venv/bin/activate | cut -d'=' -f2 | tr -d '"' )
fi

# Handle GOOGLE_CLOUD_LOCATION
if [ -n "$CURRENT_LOCATION" ]; then
    read -p "GOOGLE_CLOUD_LOCATION is currently set to '$CURRENT_LOCATION'. Overwrite? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Please enter your new Google Cloud Location (default: us-central1):"
        read location
        location=${location:-us-central1}
    else
        location=$CURRENT_LOCATION
    fi
else
    echo "Please enter your Google Cloud Location (default: us-central1):"
    read location
    location=${location:-us-central1}
fi

# Clean up old entries and add new ones
if [ -f "../venv/bin/activate" ]; then
    sed -i.bak '/export GOOGLE_CLOUD_PROJECT/d' ../venv/bin/activate
    sed -i.bak '/export GOOGLE_CLOUD_LOCATION/d' ../venv/bin/activate
    rm ../venv/bin/activate.bak* 2>/dev/null || true
fi

echo "export GOOGLE_CLOUD_PROJECT=\"$project_id\"" >> ../venv/bin/activate
echo "export GOOGLE_CLOUD_LOCATION=\"$location\"" >> ../venv/bin/activate

echo "Environment variables have been set in the virtual environment."
echo "Please restart your terminal or run 'source venv/bin/activate' to apply the changes."