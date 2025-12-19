# L30: Neural Network–based Solvers for Convection–Diffusion Systems

L30 covers the following equations and their corresponding DeepXDE notebooks:

-  [1D Steady Convection–Diffusion Equation](https://github.com/sangeeta-yadav-iisc/NIT_Hamirpur_FDP_2025/blob/main/pinntutorial1.ipynb)

- [1D Time-Dependent Burgers’ Equation](https://github.com/sangeeta-yadav-iisc/NIT_Hamirpur_FDP_2025/blob/main/burgers_equation.ipynb))


This repository accompanies **Lecture 30** on *Neural Network-based Solvers for Convection–Diffusion Systems*.
It demonstrates how **Physics-Informed Neural Networks (PINNs)** can be used to solve a 1D convection–diffusion equation.

---

## Problem Statement

We solve the steady 1D convection–diffusion equation:

$$
-\\varepsilon u''(x) + b u'(x) = f(x), \quad x \\in (0,1)
$$

with boundary conditions:

$$
u(0) = 0, \quad u(1) = 0
$$

---

## Methodology

- A fully connected neural network approximates the solution \( u(x) \)
- PDE residuals are enforced via **automatic differentiation**
- Boundary conditions are imposed through the loss function
- No mesh or numerical discretization is required

---

## What this notebook demonstrates

- Construction of a PINN using PyTorch
- Enforcement of physics constraints
- Solving convection-dominated problems
- Visualization of boundary-layer behavior

---




