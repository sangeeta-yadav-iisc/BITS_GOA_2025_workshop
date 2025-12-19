# Unified Approaches to Solving Differential Equations: From Classical Methods to Neural Networks (UniSDE-2025)

# L30: Neural Network–based Solvers for Convection–Diffusion Systems

I demonstrate the use of **Physics-Informed Neural Networks (PINNs)** with **DeepXDE** for solving following equations with corresponding notebooks:

- [1D Steady Convection–Diffusion Equation](https://github.com/sangeeta-yadav-iisc/NIT_Hamirpur_FDP_2025/blob/main/convdiff.ipynb)

- [Analytical vs PINN derivatives](https://github.com/sangeeta-yadav-iisc/NIT_Hamirpur_FDP_2025/blob/main/derivatives.ipynb)

- [1D Time-Dependent Burgers’ Equation](https://github.com/sangeeta-yadav-iisc/NIT_Hamirpur_FDP_2025/blob/main/burgers_equation.ipynb)




## Methodology

- A fully connected neural network approximates the solution $$u(x)$$
- PDE residuals are enforced via **automatic differentiation**
- Boundary conditions are imposed through the loss function
- No mesh or numerical discretization is required

---

By 

Dr. Sangeeta Yadav,

Assistant Professor,

Department of Computer Science and Engineering,

Faculty of Technology,

University of Delhi.

[Homepage](https://sites.google.com/view/sangeetayadav/home)||[Google Scholar](https://scholar.google.com/citations?user=5HYe3KwAAAAJ&hl=en) || [Linkedin](https://www.linkedin.com/in/shongi/)






