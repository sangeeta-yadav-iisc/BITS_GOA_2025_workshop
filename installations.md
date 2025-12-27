Step 1: Install Miniconda

Miniconda is a lightweight Python distribution that helps manage environments.

Download

https://docs.conda.io/en/latest/miniconda.html

Choose:

Python 3.9 or 3.10

64-bit installer

Installer suitable for your OS (Windows / macOS / Linux)

Verify Installation

Open a terminal (or Anaconda Prompt on Windows) and run:

conda --version


If a version number appears, Miniconda is installed correctly ✅

Step 2: Create a New Conda Environment

Create a clean environment for this tutorial:

conda create -n pinn_tutorial python=3.9 -y


Activate the environment:

conda activate pinn_tutorial


You should see (pinn_tutorial) in your terminal prompt.

Step 3: Upgrade pip
pip install --upgrade pip


This ensures smooth package installation.

Step 4: Install Basic Scientific Python Libraries
pip install numpy scipy matplotlib


These libraries are required for numerical computing and visualization.

Step 5: Install PyTorch (CPU Version – Recommended)
pip install torch torchvision torchaudio

Notes:

GPU is NOT required for this tutorial

CPU version is sufficient and avoids setup issues

This works on all operating systems

Step 6: Install DeepXDE
pip install deepxde


DeepXDE is the main library used for:

Defining PDEs

Computing derivatives

Implementing Physics-Informed Neural Networks

It will automatically use PyTorch as the backend.

Step 7: Install Jupyter Notebook
pip install notebook


Launch Jupyter Notebook:

jupyter notebook


Your browser should open with the Jupyter interface.

Step 8: Verify the Installation

Run the following command:

python -c "import torch, deepxde; print('PyTorch:', torch.__version__, 'DeepXDE:', deepxde.__versio