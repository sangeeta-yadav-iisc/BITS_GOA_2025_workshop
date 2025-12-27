# Unified Approaches to Solving Differential Equations: From Classical Methods to Neural Networks (UniSDE-2025)

# L30: Neural Network–based Solvers for Convection–Diffusion Systems

I demonstrate the use of **DeepXDE** for solving following excercises with corresponding notebooks:

- [1D Steady Convection–Diffusion Equation](https://github.com/sangeeta-yadav-iisc/NIT_Hamirpur_FDP_2025/blob/main/convdiff.ipynb)

- [Analytical vs PINN derivatives](https://github.com/sangeeta-yadav-iisc/NIT_Hamirpur_FDP_2025/blob/main/derivatives.ipynb)

- [1D Time-Dependent Burgers’ Equation](https://github.com/sangeeta-yadav-iisc/NIT_Hamirpur_FDP_2025/blob/main/burgers_equation.ipynb)


---

By 

Dr. Sangeeta Yadav,

Assistant Professor,

Department of Computer Science and Engineering,

Faculty of Technology,

University of Delhi.

[Homepage](https://sites.google.com/view/sangeetayadav/home)||[Google Scholar](https://scholar.google.com/citations?user=5HYe3KwAAAAJ&hl=en) || [Linkedin](https://www.linkedin.com/in/shongi/)

## Alternative: One-Command Installation

If you prefer a single command setup, use the provided install script.

### Linux / macOS / WSL
```bash
chmod +x install.sh
./install.sh
```

### Windows
install.bat



---

## Streamlit Dashboard

A Streamlit dashboard is provided in `streamlit_app.py` that reproduces the PDE setup from `burgers_equation.ipynb`, builds a PINN using DeepXDE (PyTorch backend), trains the model, and visualizes the predicted solution and loss history.

Quick start:

1. Create and activate a Python environment (recommended):

   - conda: `conda create -n unisde python=3.10 -y && conda activate unisde`
   - or venv: `python -m venv .venv && source .venv/bin/activate`

2. Install dependencies:

```
 pip install -r requirements.txt
```

Note: Installing `torch` may require selecting a platform-specific wheel. Visit https://pytorch.org for the recommended install command for macOS/CPU/GPU.

3. Run the Streamlit app:

```
 streamlit run streamlit_app.py
```

The app exposes sliders for viscosity, sampling sizes, network architecture, and training parameters. Use the "Quick demo" option for a short demo training session.







