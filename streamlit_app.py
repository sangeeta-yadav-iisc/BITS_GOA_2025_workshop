import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde

# Ensure consistent backend and float
dde.config.set_default_float("float32")
dde.backend.set_default_backend("pytorch")

# --- PDE and helper functions from the notebook ---

# default viscosity
DEFAULT_NU = 0.01 / np.pi

nu = DEFAULT_NU


def burgers_pde(x, u):
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    d2u_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t + u * du_x - nu * d2u_xx


# boundary and initial conditions (lambdas mirror those in the notebook)
bc = dde.icbc.DirichletBC(None, lambda x: 0, lambda x, on_boundary: on_boundary)

ic = dde.icbc.IC(
    None,
    lambda x: -np.sin(np.pi * x[:, 0:1]),
    lambda _, on_initial: on_initial,
)


# --- Streamlit UI ---

st.set_page_config(page_title="Burgers PINN (DeepXDE)", layout="wide")
st.title("Burgers' Equation — Physics-Informed Neural Network (PINN)")
st.markdown(
    "Use the controls on the left to configure numeric & model parameters, then train and visualize the solution." 
)

# Sidebar controls
st.sidebar.header("Simulation parameters")
nu_slider = st.sidebar.slider("Viscosity nu", min_value=1e-4, max_value=1e-1, value=float(DEFAULT_NU), format="%.6f")
num_domain = st.sidebar.number_input("Number of domain points", min_value=100, max_value=10000, value=2000, step=100)
num_boundary = st.sidebar.number_input("Number of boundary points", min_value=10, max_value=2000, value=200, step=10)
num_initial = st.sidebar.number_input("Number of initial points", min_value=10, max_value=2000, value=200, step=10)

st.sidebar.header("Neural network")
num_layers = st.sidebar.slider("Hidden layers", 1, 8, 4)
num_neurons = st.sidebar.slider("Neurons per layer", 10, 200, 50)
activation = st.sidebar.selectbox("Activation", ["tanh", "relu", "sigmoid"], index=0)
initializer = st.sidebar.selectbox("Initializer", ["Glorot normal", "He normal"], index=0)

st.sidebar.header("Training")
learning_rate = st.sidebar.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6g")
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=20000, value=1000, step=100)
quick_demo = st.sidebar.checkbox("Quick demo (short training)", value=True)

# Buttons
train_btn = st.sidebar.button("Start training")
retrain_btn = st.sidebar.button("Retrain with current settings")

# Visualization grid
st.sidebar.header("Visualization grid")
plot_x_res = st.sidebar.number_input("x resolution", min_value=50, max_value=1024, value=256, step=10)
plot_t_res = st.sidebar.number_input("t resolution", min_value=10, max_value=500, value=100, step=10)

# Placeholder for results
plot_col, info_col = st.columns([3, 1])

# Keep application state using st.session_state
if "model" not in st.session_state:
    st.session_state["model"] = None
    st.session_state["losshistory"] = None
    st.session_state["train_state"] = None


def build_problem(nu_val, ndomain, nboundary, ninitial):
    global nu
    nu = nu_val
    geom = dde.geometry.Interval(-1, 1)
    time_domain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, time_domain)

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary)

    ic = dde.icbc.IC(
        geomtime,
        lambda x: -np.sin(np.pi * x[:, 0:1]),
        lambda _, on_initial: on_initial,
    )

    data = dde.data.TimePDE(
        geomtime,
        burgers_pde,
        [bc, ic],
        num_domain=ndomain,
        num_boundary=nboundary,
        num_initial=ninitial,
    )
    return data


def build_model(data, nlayers, nneurons, act, init):
    layer_sizes = [2] + [int(nneurons)] * int(nlayers) + [1]
    net = dde.nn.FNN(layer_sizes, act, init)
    model = dde.Model(data, net)
    return model


def train_model(model, lr, n_epochs, quick=False):
    model.compile("adam", lr=lr)
    if quick:
        # short run using fewer epochs for quick feedback
        e = min(200, n_epochs)
    else:
        e = n_epochs
    losshistory, train_state = model.train(epochs=int(e))
    return losshistory, train_state


def plot_solution(model, x_res=256, t_res=100):
    x = np.linspace(-1, 1, x_res)
    t = np.linspace(0, 1, t_res)
    X, T = np.meshgrid(x, t)
    XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_pred = model.predict(XT).reshape(T.shape)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(u_pred, extent=[-1, 1, 0, 1], origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax, label='u(x,t)')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title("Burgers' Equation Solution (PINN)")
    return fig, u_pred


# Actions
if train_btn or retrain_btn:
    with st.spinner("Building problem and training model — this may take a while..."):
        data = build_problem(nu_slider, int(num_domain), int(num_boundary), int(num_initial))
        model = build_model(data, num_layers, num_neurons, activation, initializer)
        losshistory, train_state = train_model(model, learning_rate, int(epochs), quick=quick_demo)

        st.session_state["model"] = model
        st.session_state["losshistory"] = losshistory
        st.session_state["train_state"] = train_state

        st.success("Training finished")

# Display model and plots if available
if st.session_state["model"] is not None:
    fig, u_pred = plot_solution(st.session_state["model"], x_res=int(plot_x_res), t_res=int(plot_t_res))
    plot_col.pyplot(fig)

    # Loss history
    if st.session_state["losshistory"] is not None:
        loss_train = np.array(st.session_state["losshistory"].loss_train)
        fig2, ax2 = plt.subplots()
        ax2.semilogy(loss_train)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Training loss")
        ax2.set_title("Loss history (log scale)")
        plot_col.pyplot(fig2)

    info_col.subheader("Model info")
    info_col.write(f"Layers: {num_layers}, Neurons: {num_neurons}")
    info_col.write(f"Activation: {activation}")
    info_col.write(f"Epochs (requested): {epochs}")
    if st.session_state["losshistory"] is not None:
        info_col.write(f"Final loss: {st.session_state['losshistory'].loss_train[-1]:.3e}")

else:
    st.info("No trained model yet — use the left sidebar controls and press 'Start training' to run a demo training.")

# Provide option to export prediction as npy
if st.session_state["model"] is not None:
    if st.button("Download predicted solution (npy)"):
        _, u_pred = plot_solution(st.session_state["model"], x_res=int(plot_x_res), t_res=int(plot_t_res))
        np.save("u_pred.npy", u_pred)
        with open("u_pred.npy", "rb") as f:
            st.download_button("Download u_pred.npy", data=f, file_name="u_pred.npy")


# Footer
st.markdown("---")
st.markdown("**Notes:** This app uses DeepXDE with PyTorch backend. Training can be slow — use the Quick demo option for a fast run.")