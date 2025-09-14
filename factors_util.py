import pandas as pd
import statsmodels.api as sm
import pandas_datareader.data as web
import datetime
import yfinance as yf
from statsmodels import regression
from sklearn.linear_model import LassoCV
import numpy as np
from sklearn.preprocessing import StandardScaler
from textwrap import fill

def load_ff_factors(file_name: str) -> pd.DataFrame:
    """
    Download a factors dataset from Ken French dataset into a CSV file
    Converts returns to numbers from 0 and 1 (from percent values)
    Converts the index to monthly periods
    """
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{file_name}"
    
    # Read raw text first to figure out where the monthly table ends
    raw = pd.read_csv(url, skiprows=3)

    # Find the row where 'Annual Factors' starts
    stop_row = raw[raw.iloc[:,0].astype(str).str.contains("Annual", na=False)].index[0]

    # Now re-read only up to that line
    ff = pd.read_csv(url, skiprows=3, nrows=stop_row, index_col=0)

    # Clean up: convert index to datetime
    ff.index = pd.to_datetime(ff.index, format="%Y%m").to_period('M')
    ff = ff.apply(pd.to_numeric, errors="coerce") / 100.0   # convert to decimals
    return ff

def download_from_yf(ticker: str, start_date: str, end_date: str, field_name: str = 'Close'):
    """
    Downloads returns from Yhaoo finance for a ticker and start end end dates
    Returns a data frame with columns Date (Monthly period) and Portfolio (returns)
    """
    df_series = yf.download(ticker, 
                            start=start_date, 
                            end=end_date,
                            interval="1mo", 
                            auto_adjust=True)[field_name]
    df_series = df_series.pct_change().dropna()
    portfolio = pd.DataFrame(df_series)
    portfolio.columns = ["Portfolio"]
    portfolio.index = portfolio.index.to_period('M')
    return portfolio

def annualize_alpha(alpha_monthly: float) -> float:
    return (1 + alpha_monthly)**12 - 1

def factor_model_statsmodel(
    portfolio: pd.DataFrame, 
    factor_returns: pd.DataFrame, 
    factor_columns: list[str] = None,
    risk_free_column: str = "RF"
) -> regression.linear_model.RegressionResultsWrapper:
    """
    Computes and returns a regression model of the portfolio by factors
    """
    fac_columns = factor_columns
    if fac_columns is None:
        fac_columns = [col for col in factor_returns.columns if col != risk_free_column]
    data = portfolio.join(factor_returns, how="inner")
    
    # 4. Excess portfolio return
    data["Excess_Portfolio"] = data["Portfolio"] - data[risk_free_column]
    
    # 5. Regression: Excess_Portfolio ~ Mkt-RF + SMB + HML
    X = data[fac_columns]
    y = data["Excess_Portfolio"]
    X = sm.add_constant(X)  # adds alpha term
    
    model = sm.OLS(y, X).fit()
    data['replicating'] = model.fittedvalues
    data['alpha'] = data['Excess_Portfolio'] - data['replicating']
    data['replicating_total'] = data['replicating'] + data[risk_free_column]
    # cumulative = (1 + data[["Excess_Portfolio","replicating","alpha"]]).cumprod()
    cumulative = pd.DataFrame({
        "actual":        (1 + data["Portfolio"]).cumprod(),
        "replicating":   (1 + data["replicating_total"]).cumprod(),
        "alpha (residual)": (1 + data["alpha"]).cumprod(),
    })
    return model, data, cumulative

import matplotlib.pyplot as plt

def get_formula_and_rsq(model: regression.linear_model.RegressionResultsWrapper):
    alpha = model.params["const"]
    betas = model.params.drop("const")
    
    # Format equation: y = alpha + b1*X1 + b2*X2 + ...
    formula = f"y = {alpha:.4f}"
    for name, coef in betas.items():
        formula += f"\n + {coef:.4f}·{name}"
    
    rsq = model.rsquared
    return formula, rsq

def put_formula_inside_top(ax, formula, fontsize=10, color="darkblue"):
    # wrap but keep any manual \n the user included
    fig = ax.figure
    fig.canvas.draw()  # need renderer for size info
    ax_w_px = ax.get_window_extent().width
    avg_char_px = 0.6 * fontsize
    width = max(12, int(ax_w_px / avg_char_px))
    wrapped = "\n".join(fill(line, width=width) for line in formula.splitlines())

    ax.text(
        0.5, 0.99, wrapped,                 # just inside the top of the axes
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=fontsize, color=color,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.3),
    )

    
def plot_model_statsmodel(
    cumulative: pd.DataFrame,
    model: regression.linear_model.RegressionResultsWrapper,
    plot_title: str = "Portfolio vs Replicating Factor Model vs Alpha"
):
    """
    The input cumulative dataframe should have cumulative returns,
    and the index should be periods
    """
    x_index = cumulative.index.to_timestamp()
    formula, rsq = get_formula_and_rsq(model)
    alpha_ann = annualize_alpha(model.params["const"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_index, cumulative["actual"], label="actual")
    ax.plot(x_index, cumulative["replicating"], label="replicating")
    ax.plot(x_index, cumulative["alpha (residual)"], label="alpha (residual)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth (Starting at 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Normal title inside axes space
    # ax.set_title(plot_title, fontsize=14, pad=12)
    ax.set_title(plot_title + f"\n(R² = {rsq:.3f}, α ≈ {alpha_ann:.2%}/yr)", pad=10, fontsize=14)
    
    # Put formula just under the title, but above the chart
    # plt.text(0.5, 0.8, formula,
    #          ha="center", va="bottom",
    #          transform=plt.gca().transAxes,
    #          fontsize=10, color="darkblue")
    put_formula_inside_top(ax, formula, fontsize=10)
   
    # R² annotation near x-axis
    plt.text(cumulative.index[int(len(cumulative)*0.4)], 
             cumulative.min().min()*0.95, 
             f"R² = {rsq:.3f}", 
             fontsize=10, color="darkred")
    
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth (Starting at 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def factor_model_lasso_old(
    portfolio: pd.DataFrame, 
    factor_returns: pd.DataFrame, 
    factor_columns: list[str] = None,
    risk_free_column: str = "RF"
) -> tuple[LassoCV, pd.Series, float, pd.DataFrame, pd.DataFrame]:
    """
    Computes and returns a regression model of the portfolio by factors
    Uses the lasso method for regression analysis
    """
    fac_columns = factor_columns
    if fac_columns is None:
        fac_columns = [col for col in factor_returns.columns if col != risk_free_column]
    data = portfolio.join(factor_returns, how="inner")
    
    # 4. Excess portfolio return
    data["Excess_Portfolio"] = data["Portfolio"] - data[risk_free_column]
    
    X = data[fac_columns]
    y = data["Excess_Portfolio"].values
    
    # Standardize features (important for Lasso penalties)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Lasso with cross-validation to pick alpha (penalty strength)
    lasso = LassoCV(cv=5).fit(X_scaled, y)
    
    # Get coefficients back in original scale
    coefs = pd.Series(lasso.coef_, index=X.columns)
    
    # print("Intercept (alpha):", lasso.intercept_)
    # print("Best penalty α:", lasso.alpha_)
    rsq = lasso.score(X_scaled, y)
    # print("R²:", r2)
    y_hat = lasso.predict(X_scaled)
    data["Replicating_Lasso"] = y_hat
    data["Alpha_Lasso"] = data["Excess_Portfolio"] - data["Replicating_Lasso"]
    cumulative = (1 + data[["Excess_Portfolio","Replicating_Lasso","Alpha_Lasso"]]).cumprod()

    return lasso, coefs, rsq, data, cumulative 

def factor_model_lasso(
    portfolio: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_columns: list[str] | None = None,
    risk_free_column: str = "RF",
    use_1se_rule: bool = False,        # pick sparser λ if True
    cv: int = 5,
) -> tuple[LassoCV, pd.Series, float, float, pd.DataFrame, pd.DataFrame]:
    """
    Lasso regression of portfolio excess returns on factors.

    Returns:
      lasso (fitted estimator on standardized X),
      coefs (Series of betas in original units),
      intercept (alpha in original units),
      rsq (R^2 in-sample on training data),
      data (joined DataFrame with columns: Excess_Portfolio, Replicating_Lasso, Alpha_Lasso),
      cumulative (cumprod of 1 + those three series)
    """
    df = portfolio.join(factor_returns, how="inner").copy()

    # Decide factor columns (exclude RF)
    fac_cols = factor_columns or [c for c in factor_returns.columns if c != risk_free_column]

    # Excess portfolio return (make sure portfolio is also in decimals)
    df["Excess_Portfolio"] = df["Portfolio"] - df[risk_free_column]

    X = df[fac_cols].astype(float).to_numpy()
    y = df["Excess_Portfolio"].astype(float).to_numpy()

    # Standardize X for L1 penalty
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Fit LassoCV
    lcv = LassoCV(cv=cv, alphas=200, random_state=0).fit(Xs, y)

    # Optional: 1-SE rule for a sparser model
    if use_1se_rule:
        # mse_path_: shape (n_alphas, n_folds); alphas_ decreasing
        mse_mean = lcv.mse_path_.mean(axis=1)
        mse_std = lcv.mse_path_.std(axis=1, ddof=1)
        min_idx = mse_mean.argmin()
        thresh = mse_mean[min_idx] + mse_std[min_idx]    # 1-SE threshold
        # pick the LARGEST alpha whose mean MSE <= threshold (sparser)
        idx_1se = np.where(mse_mean <= thresh)[0][0]
        alpha_1se = lcv.alphas_[idx_1se]
        # refit at alpha_1se on same standardized X
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=alpha_1se, fit_intercept=True).fit(Xs, y)
    else:
        lasso = lcv

    # ---- Back-transform coefficients to original feature units ----
    # lasso.coef_ are for standardized X
    beta_std = lasso.coef_
    sigma = scaler.scale_
    mu = scaler.mean_

    beta = beta_std / sigma
    intercept = lasso.intercept_ - np.dot(beta, mu)

    coefs = pd.Series(beta, index=fac_cols)

    # In-sample R^2 on training data
    rsq = lasso.score(Xs, y)

    # Predictions in returns space
    y_hat = lasso.predict(Xs)  # same as intercept + (X - mu) @ (beta_std)
    df["Replicating_Lasso"] = y_hat
    
    # df["Alpha_Lasso"] = df["Excess_Portfolio"] - df["Replicating_Lasso"]
    # cumulative = (1 + df[["Excess_Portfolio", "Replicating_Lasso", "Alpha_Lasso"]]).cumprod()

    df["Alpha_Lasso"] = df["Excess_Portfolio"] - y_hat
    
    # total returns
    df["Portfolio_Total"]   = df["Portfolio"]          # already total (decimal)
    df["Replicating_Total"] = df["Replicating_Lasso"] + df[risk_free_column]
    df["Alpha_Total"]       = df["Alpha_Lasso"]        # alpha stays excess
    
    # cumulative growth (start at 1)
    cumulative = pd.DataFrame({
        "actual":        (1 + df["Portfolio_Total"]).cumprod(),
        "replicating":   (1 + df["Replicating_Total"]).cumprod(),
        "alpha (residual)": (1 + df["Alpha_Total"]).cumprod(),
    })

    return lcv, coefs, intercept, rsq, df, cumulative

def sign(val):
    if val >= 0:
        return "+"
    return "-"
    
def get_formula_lasso_old(lasso, coefs) -> str:
    # Build regression equation string from model parameters
    alpha = lasso.intercept_
    betas = coefs
    
    # Format equation: y = alpha + b1*X1 + b2*X2 + ...
    formula = f"y = {alpha:.4f}"
    for name, coef in betas.items():
        formula += f"\n {sign(coef)} {abs(coef):.4f}·{name}"
    return formula

def get_formula_lasso(
    intercept: float,
    coefs: pd.Series,
    *,
    sort_by_abs: bool = True,   # show largest loadings first
    zero_tol: float = 5e-5,     # hide “effectively zero” betas
    coef_fmt: str = "{:+.4f}",  # sign + 4 decimals
    wrap_chars: int | None = 80 # soft wrap long lines; None = no wrap
) -> str:
    """
    Build a readable multiline formula string like:

      y = 0.0094
       + 0.0459·Mkt-RF
       - 0.0047·SMB
       + 0.0019·HML
       + 0.0011·RMW

    Args:
      intercept: alpha in *original return units* (e.g., monthly decimal)
      coefs: pd.Series of betas in original units, index = factor names
    """
    # Optionally drop near-zeros for cleaner display
    show = coefs.copy()
    show[show.abs() < zero_tol] = 0.0

    if sort_by_abs:
        show = show.reindex(show.abs().sort_values(ascending=False).index)

    lines = [f"y = {intercept:.4f}"]
    for name, beta in show.items():
        if beta == 0.0:  # skip if zero after tolerance
            continue
        # U+00B7 is a middle dot
        lines.append(f"{coef_fmt.format(beta)}·{name}")

    # If everything got zeroed, still show one zero term (optional)
    if len(lines) == 1:
        # keep the largest original (even if tiny) so the user sees structure
        name = coefs.abs().idxmax()
        lines.append(f"{coef_fmt.format(coefs[name])}·{name}")

    s = "\n ".join([lines[0]] + lines[1:])  # place '+'/'-' on each line start

    if wrap_chars:
        # wrap each existing line separately to keep manual newlines
        s = "\n".join(fill(line, width=wrap_chars) for line in s.splitlines())

    return s

def plot_model_lasso_old(
    cumulative: pd.DataFrame,
    intercept: float,
    coefs: pd.Series,
    rsq: float,
    plot_title: str = "Portfolio vs Replicating Factor Model vs Alpha"
):
    x_index = cumulative.index.to_timestamp()
    formula = get_formula_lasso(intercept, coefs)
    
    plt.figure(figsize=(12,6))
    plt.plot(x_index, cumulative["Excess_Portfolio"], label="actual")
    plt.plot(x_index, cumulative["Replicating_Lasso"], label="replicating")
    plt.plot(x_index, cumulative["Alpha_Lasso"], label="alpha (residual)")
    
    # Main title = regression formula
    plt.title(plot_title, fontsize=14)
    
    # Put formula just under the title, but above the chart
    plt.text(0.5, 0.8, formula,
             ha="center", va="bottom",
             transform=plt.gca().transAxes,
             fontsize=10, color="darkblue")
    
    # R² annotation near x-axis
    plt.text(cumulative.index[int(len(cumulative)*0.4)], 
             cumulative.min().min()*0.95, 
             f"R² = {rsq:.3f}", 
             fontsize=10, color="darkred")
    
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth (Starting at 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

import matplotlib.pyplot as plt
from textwrap import fill

def plot_model_lasso_old_v2(cumulative, lasso, coefs, rsq,
                     plot_title="Portfolio vs Replicating Factor Model vs Alpha"):
    x_index = cumulative.index.to_timestamp()
    formula = get_formula_lasso(lasso, coefs)

    # ---- base plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_index, cumulative["Excess_Portfolio"], label="actual")
    ax.plot(x_index, cumulative["Replicating_Lasso"], label="replicating")
    ax.plot(x_index, cumulative["Alpha_Lasso"], label="alpha (residual)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth (Starting at 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- title placed as suptitle (above axes) ----
    t = fig.suptitle(plot_title, fontsize=14, y=1.0, va="bottom")

    # ---- wrap the formula to the axis width & measure its height ----
    fig.canvas.draw()  # ensure a renderer exists

    ax_w_px = ax.get_window_extent().width
    fontsize = 10
    # crude avg char width ≈ 0.6 * fontsize (pixels)
    # chars_per_line = max(12, int(ax_w_px / (0.6 * fontsize)))
    # formula_wrapped = _wrap_preserve_newlines(formula, chars_per_line)

    # put formula at the very top temporarily to measure it
    txt = fig.text(0.5, 1.0, formula, ha="center", va="bottom",
                   fontsize=fontsize, color="darkblue")
    fig.canvas.draw()
    bb = txt.get_window_extent(fig.canvas.get_renderer())
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    pad_px = 6
    band_h_frac = (bb.height + pad_px) / fig_h_px

    # now reserve space for BOTH title and formula bands
    # (the suptitle also takes space; use a small extra buffer)
    title_pad_frac = 0.02
    top_margin = 1 - (band_h_frac + title_pad_frac)
    if top_margin < 0.75:  # cap so we don't squash the plot too much
        top_margin = 0.75
    plt.subplots_adjust(top=top_margin)

    # reposition the formula just above the axes, under the suptitle
    txt.set_position((0.5, top_margin + 0.002))

    # ---- R²: anchor using axes coords so it stays near the x-axis ----
    ax.text(0.5, -0.12, f"R² = {rsq:.3f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color="darkred")

    plt.show()

def draw_formula_top(fig, formula, fontsize=10, color="darkblue", pad=4):
    """
    Draw multiline formula at the top of the figure,
    auto-adjusting Y so it never collides with the title or goes off the page.
    Returns the text artist.
    """
    # Optional: wrap long lines to avoid spilling horizontally
    chars_per_line = 80  # tweak or compute from figure width
    formula_wrapped = "\n".join(fill(line, width=chars_per_line)
                                for line in formula.splitlines())

    # First draw at y=1.0 to measure size
    txt = fig.text(0.5, 1.0, formula_wrapped,
                   ha="center", va="top", fontsize=fontsize, color=color)
    fig.canvas.draw()
    bb = txt.get_window_extent(fig.canvas.get_renderer())

    # Convert height in pixels → figure fraction
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    h_frac = bb.height / fig_h_px

    # Place properly, leaving small pad
    y_pos = 1.0 - pad / fig_h_px
    txt.set_position((0.5, y_pos))

    return txt, h_frac

def plot_model_lasso(cumulative, intercept, coefs, rsq,
                     plot_title="Portfolio vs Replicating Factor Model vs Alpha"):
    x_index = cumulative.index.to_timestamp()
    formula = get_formula_lasso(intercept, coefs)
    alpha_ann = annualize_alpha(intercept)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_index, cumulative["actual"], label="actual")
    ax.plot(x_index, cumulative["replicating"], label="replicating")
    ax.plot(x_index, cumulative["alpha (residual)"], label="alpha (residual)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth (Starting at 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Normal title inside axes space
    # ax.set_title(plot_title, fontsize=14, pad=12)
    ax.set_title(plot_title + f"\n(R² = {rsq:.3f}, α ≈ {alpha_ann:.2%}/yr)", pad=10, fontsize=14)

    # Formula at very top of figure
    # fig.text(0.5, 0.85, formula,
    #          ha="center", va="top",
    #          fontsize=10, color="darkblue")
    # Draw formula above everything, with auto height handling
    # draw_formula_top(fig, formula, fontsize=10, color="darkblue")
    
    # Formula sits INSIDE the plot at the top; long formulas wrap and extend downward
    put_formula_inside_top(ax, formula, fontsize=10)

    # R² near bottom axis
    # ax.text(0.5, -0.12, f"R² = {rsq:.3f}", transform=ax.transAxes,
    #         ha="center", va="top", fontsize=10, color="darkred")
    # R² annotation near x-axis
    plt.text(cumulative.index[int(len(cumulative)*0.4)], 
             cumulative.min().min()*0.95, 
             f"R² = {rsq:.3f}", 
             fontsize=10, color="darkred")

    plt.show()
