import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# Utilities
# -----------------------------
def parse_csv_numbers(s):
    if not s.strip():
        return np.array([])
    return np.array([float(x.strip()) for x in s.split(",")])

def standardize(arr):
    mu = np.mean(arr)
    sigma = np.std(arr)
    sigma = sigma if sigma > 0 else 1.0
    return (arr - mu) / sigma, mu, sigma

def destandardize_params(m_z, b_z, x_mu, x_sd, y_mu, y_sd):
    """
    Convert params learned in standardized space (z-scores) back to original scale:
      y = (y_sd * m_z / x_sd) * x + (y_sd * (b_z - m_z * x_mu / x_sd) + y_mu)
    Returns (m_orig, b_orig)
    """
    m_orig = (y_sd * m_z) / x_sd
    b_orig = y_sd * (b_z - m_z * x_mu / x_sd) + y_mu
    return m_orig, b_orig

def mse(y, yhat):
    return np.mean((y - yhat) ** 2)

def batch_gradients(x, y, m, b):
    """
    MSE gradients:
      dL/dm = (-2/n) * sum( x_i * (y_i - (m x_i + b)) )
      dL/db = (-2/n) * sum(      (y_i - (m x_i + b)) )
    """
    n = x.size
    yhat = m * x + b
    resid = y - yhat
    dm = (-2.0 / n) * np.sum(x * resid)
    db = (-2.0 / n) * np.sum(resid)
    loss = mse(y, yhat)
    return dm, db, loss

def closed_form_solution(x, y):
    """
    For simple linear regression with intercept:
      m* = cov(x,y)/var(x)
      b* = mean(y) - m* mean(x)
    Handles var(x)=0 by returning m=0 and b=mean(y)
    """
    xmu, ymu = np.mean(x), np.mean(y)
    x_var = np.var(x)
    if x_var <= 1e-15:
        return 0.0, float(ymu)
    cov_xy = np.mean((x - xmu) * (y - ymu))
    m_star = cov_xy / x_var
    b_star = ymu - m_star * xmu
    return float(m_star), float(b_star)

def run_gradient_descent(
    x, y, lr, epochs, m0, b0, standardize_flag, tol, clip_val, max_history
):
    """
    Runs full-batch GD, optionally in standardized space, and returns a dict of results.
    """
    # Possibly standardize
    if standardize_flag:
        x_z, x_mu, x_sd = standardize(x)
        y_z, y_mu, y_sd = standardize(y)
        x_work, y_work = x_z, y_z
        # initialize in standardized space
        m, b = m0, b0
    else:
        x_work, y_work = x, y
        x_mu = x_sd = y_mu = y_sd = None
        m, b = m0, b0

    m_hist, b_hist, loss_hist = [m], [b], []

    for t in range(epochs):
        dm, db, loss = batch_gradients(x_work, y_work, m, b)

        # Early stop on small gradient magnitude
        if np.hypot(dm, db) < tol:
            loss_hist.append(loss)
            break

        # Parameter update
        m = m - lr * dm
        b = b - lr * db

        # Optional clipping to avoid overflow for bad settings
        if clip_val is not None:
            m = float(np.clip(m, -clip_val, clip_val))
            b = float(np.clip(b, -clip_val, clip_val))

        # Record history (truncate for memory)
        m_hist.append(m)
        b_hist.append(b)
        loss_hist.append(loss)
        if len(m_hist) > max_history:
            m_hist = m_hist[-max_history:]
            b_hist = b_hist[-max_history:]
            loss_hist = loss_hist[-max_history:]

    # Final values
    m_final, b_final = m, b

    # Map back to original space if standardized
    if standardize_flag:
        m_orig, b_orig = destandardize_params(m_final, b_final, x_mu, x_sd, y_mu, y_sd)
        # Also compute path in original space for plotting line
        m_path = []
        b_path = []
        for m_t, b_t in zip(m_hist, b_hist):
            mo, bo = destandardize_params(m_t, b_t, x_mu, x_sd, y_mu, y_sd)
            m_path.append(mo)
            b_path.append(bo)
    else:
        m_orig, b_orig = m_final, b_final
        m_path, b_path = m_hist, b_hist

    return {
        "m_final": m_final,
        "b_final": b_final,
        "loss_hist": loss_hist,
        "m_hist": m_hist,
        "b_hist": b_hist,
        "m_orig": m_orig,
        "b_orig": b_orig,
        "m_path": m_path,
        "b_path": b_path,
        "standardize": standardize_flag,
    }

def make_loss_surface(x, y, m_star, b_star, span_m=3.0, span_b=3.0, steps=120):
    """
    Build a grid of MSE(m,b) around (m*, b*), for contour plotting.
    """
    # pick ranges
    m_min, m_max = m_star - span_m, m_star + span_m
    b_min, b_max = b_star - span_b, b_star + span_b

    M = np.linspace(m_min, m_max, steps)
    B = np.linspace(b_min, b_max, steps)
    MM, BB = np.meshgrid(M, B)

    # Broadcast to compute MSE over grid
    # Shape: (steps, steps, n)
    yhat_grid = MM[..., None] * x[None, None, :] + BB[..., None]
    Z = np.mean((y[None, None, :] - yhat_grid) ** 2, axis=2)
    return M, B, Z

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Interactive Gradient Descent (Linear Regression)", layout="wide")
st.title("📉 Interactive Gradient Descent — Linear Regression")

with st.sidebar:
    st.header("Data")
    default_x = "20, 30, 50"
    default_y = "3, 4, 5"
    x_str = st.text_input("x values (comma-separated):", value=default_x)
    y_str = st.text_input("y values (comma-separated):", value=default_y)
    x = parse_csv_numbers(x_str)
    y = parse_csv_numbers(y_str)

    if x.size != y.size or x.size == 0:
        st.error("x and y must be non-empty and have the same length.")
        st.stop()

    st.header("Gradient Descent Settings")
    lr = st.number_input("Learning rate (α)", value=1e-3, min_value=1e-9, max_value=1.0, step=1e-3, format="%.8f")
    epochs = st.number_input("Max epochs", value=2000, min_value=1, max_value=200000)
    m0 = st.number_input("Init slope m₀", value=0.0, format="%.6f")
    b0 = st.number_input("Init intercept b₀", value=0.0, format="%.6f")
    standardize_flag = st.checkbox("Standardize x and y (recommended)", value=True)
    tol = st.number_input("Early-stop gradient tolerance", value=1e-8, min_value=0.0, format="%.1e")
    clip = st.checkbox("Clip parameters to avoid overflow", value=True)
    clip_val = 1e6 if clip else None

    st.header("Surface Plot Settings")
    span_m = st.slider("m-span around optimum", 0.1, 10.0, 3.0, 0.1)
    span_b = st.slider("b-span around optimum", 0.1, 10.0, 3.0, 0.1)
    steps = st.slider("Grid resolution (higher = smoother, slower)", 40, 200, 120, 10)

# Closed-form baseline (original space)
m_star, b_star = closed_form_solution(x, y)
y_pred_star = m_star * x + b_star
loss_star = mse(y, y_pred_star)

# Run GD
result = run_gradient_descent(
    x=x,
    y=y,
    lr=lr,
    epochs=int(epochs),
    m0=m0,
    b0=b0,
    standardize_flag=standardize_flag,
    tol=tol,
    clip_val=clip_val,
    max_history=10000,
)

# Final model in original space
m_final_orig = result["m_orig"]
b_final_orig = result["b_orig"]
y_pred_final = m_final_orig * x + b_final_orig
final_loss = mse(y, y_pred_final)

# -----------------------------
# Top metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Closed-form m*", f"{m_star:.6f}")
c2.metric("Closed-form b*", f"{b_star:.6f}")
c3.metric("Final GD m", f"{m_final_orig:.6f}")
c4.metric("Final GD b", f"{b_final_orig:.6f}")

c5, c6 = st.columns(2)
c5.metric("Closed-form MSE", f"{loss_star:.6e}")
c6.metric("Final GD MSE", f"{final_loss:.6e}")

# -----------------------------
# Plots
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Data & Fit", "Loss vs Epoch", "Loss Surface + Path"])

with tab1:
    st.subheader("Data and regression line")
    xs = np.linspace(np.min(x), np.max(x), 200)
    ys_final = m_final_orig * xs + b_final_orig
    ys_star = m_star * xs + b_star

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=xs, y=ys_final, mode="lines", name="GD line"))
    fig.add_trace(go.Scatter(x=xs, y=ys_star, mode="lines", name="Closed-form line", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="x", yaxis_title="y", height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("MSE over iterations")
    loss_hist = result["loss_hist"]
    if len(loss_hist) == 0:
        st.info("No iterations recorded (possibly converged immediately).")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=loss_hist, mode="lines", name="MSE"))
        fig2.update_layout(xaxis_title="epoch", yaxis_title="MSE", height=450)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("MSE Contour (original space) with GD path")
    M, B, Z = make_loss_surface(x, y, m_star, b_star, span_m=span_m, span_b=span_b, steps=steps)

    fig3 = go.Figure(
        data=go.Contour(
            x=M, y=B, z=Z.T, contours_coloring="heatmap", showscale=True, ncontours=40
        )
    )
    # GD path in original space
    fig3.add_trace(
        go.Scatter(
            x=result["m_path"], y=result["b_path"],
            mode="markers+lines", name="GD path", marker=dict(size=6)
        )
    )
    # Mark closed-form optimum
    fig3.add_trace(
        go.Scatter(
            x=[m_star], y=[b_star],
            mode="markers", name="Closed-form optimum",
            marker=dict(size=10, symbol="x")
        )
    )
    fig3.update_layout(xaxis_title="m (slope)", yaxis_title="b (intercept)", height=600)
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# Debug / details
# -----------------------------
with st.expander("Details & last gradients"):
    st.write("**Standardize on?**", result["standardize"])
    # Recompute last gradients in the *current working space*, for transparency
    if result["standardize"]:
        # Recreate working arrays and last m,b to compute grads in that space
        x_z, x_mu, x_sd = standardize(x)
        y_z, y_mu, y_sd = standardize(y)
        m_last = result["m_hist"][-1]
        b_last = result["b_hist"][-1]
        dm, db, loss_work = batch_gradients(x_z, y_z, m_last, b_last)
    else:
        m_last = result["m_hist"][-1]
        b_last = result["b_hist"][-1]
        dm, db, loss_work = batch_gradients(x, y, m_last, b_last)

    st.write(f"Last grads (working space):  dL/dm = {dm:.6e},  dL/db = {db:.6e}")
    st.write(f"Working-space MSE at last step: {loss_work:.6e}")
    st.write("History length:", len(result["loss_hist"]))
