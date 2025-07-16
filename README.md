# Model_Portfolio
i am working on one project that select top 30 companies according to given companies data and industry data. then i provide my own view after which i optimise it to find weight of stocks.

import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize, LinearConstraint

df = pd.read_excel('/content/Data_To_analyse.xlsx')
df.head()
# Clean column names for consistency
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
# print(df.columns)

# Filter By Total Market Cap
def filter_by_total_market_cap(df, min_cap):
    filtered_df = df[(df['market_cap'] >= min_cap)]
    print(f"{len(filtered_df)} stocks passed Market Cap >= {min_cap}")
    return filtered_df

# Filter By Recommendation
def filter_by_recommendation(df, accepted = ['Buy' , 'Hold']):
    filtered_df = df[df['recommendation'].isin(accepted)]
    print(f"{len(filtered_df)} stocks passed Recommendation ({' or '.join(accepted)})")
    return filtered_df

# Filter By Upside/Downside
def filter_by_upside_downside(df, min_upside):
    filtered_df = df[df['upside/downside'] >= min_upside]
    print(f"{len(filtered_df)} stocks passed Upside >= {min_upside}")
    return filtered_df

# Filter By Revenue CAGR
def filter_by_Revenue_Cagr(df, min_revenue_cagr):
    filtered_df = df[df['revenue_cagr'] >= min_revenue_cagr]
    print(f"{len(filtered_df)} stocks passed Revenue CAGR >= {min_revenue_cagr}")
    return filtered_df

# Filter By ROCE FY27
def filter_by_ROCE_FY27E(df, min_roce):
    filtered_df = df[df['roce_fy27e'] >= min_roce]
    print(f"{len(filtered_df)} stocks passed ROCE >= {min_roce}")
    return filtered_df

def filter_by_ROE(df, min_roe):
    filtered_df = df[df['roe_fy27e'] >= min_roe]
    print(f"{len(filtered_df)} stocks passed ROE >= {min_roe}")
    return filtered_df

# Filter By Debt to Equity
def filter_by_debt_to_equity(df, min_debt_to_equity):
    filtered_df = df[df['debt_to_equity'] >= min_debt_to_equity]
    print(f"{len(filtered_df)} stocks passed Debt to Equity >= {min_debt_to_equity}")
    return filtered_df

# Filter By EPS CAGR
def filter_by_EPS_CAGR(df, min_eps_cagr):
    filtered_df = df[df['eps_cagr'] >= min_eps_cagr]
    print(f"{len(filtered_df)} stocks passed EPS CAGR >= {min_eps_cagr}")
    return filtered_df


# Apply Total Market Cap Filter

min_market_cap = 2

filtered_df_market_cap = filter_by_total_market_cap(df, min_market_cap)
filtered_df_market_cap.info()

# Apply Filter By Recommendation
filtered_df_recommendation = filter_by_recommendation(filtered_df_market_cap)
filtered_df_recommendation.info()

# Apply Filter By Upside/Downside
min_upside = 0.1 # Minimum upside
filtered_df_upside = filter_by_upside_downside(filtered_df_recommendation, min_upside)
filtered_df_upside.info()
filtered_df_upside = filtered_df_upside.copy()

# Convert relevant columns to numeric to avoid TypeErrors
filtered_df_upside['eps_cagr'] = pd.to_numeric(filtered_df_upside['eps_cagr'], errors='coerce')
filtered_df_upside['revenue_cagr'] = pd.to_numeric(filtered_df_upside['revenue_cagr'], errors='coerce')
filtered_df_upside['roce_fy27e'] = pd.to_numeric(filtered_df_upside['roce_fy27e'], errors='coerce')
filtered_df_upside['debt_to_equity'] = pd.to_numeric(filtered_df_upside['debt_to_equity'], errors='coerce')
filtered_df_upside['roe_fy27e'] = pd.to_numeric(filtered_df_upside['roe_fy27e'], errors='coerce')

# Define Scoring Criteria
def score_EPS_CAGR(eps_cagr):
    if eps_cagr >= 50: return 5
    elif eps_cagr >= 40: return 4
    elif eps_cagr >= 30: return 3
    elif eps_cagr >= 20: return 2
    else: return 0

def score_Revenue_CAGR(revenue_cagr):
    if revenue_cagr >= 50: return 5
    elif revenue_cagr >= 30: return 4
    elif revenue_cagr >= 20: return 3
    elif revenue_cagr >= 10: return 2
    else: return 0

def score_ROCE_FY27E(roce_fy27e):
    if roce_fy27e >= 30: return 5
    elif roce_fy27e >= 25: return 4
    elif roce_fy27e >= 20: return 3
    elif roce_fy27e >= 15: return 2
    else: return 0

def score_Debt_to_Equity(debt_to_equity):
    if debt_to_equity <= 1: return 5
    elif debt_to_equity <= 1.5: return 4
    elif debt_to_equity <= 2: return 3
    elif debt_to_equity <= 3: return 2
    else: return 0

def score_ROE(roe):
    if roe >= 30: return 5
    elif roe >= 25: return 4
    elif roe >= 20: return 3
    elif roe >= 15: return 2
    else: return 0

# Assign Weights
weights_of_parameter = {
    'EPS_CAGR': 0.3,
    'Revenue_CAGR': 0.2,
    'ROCE_FY27E': 0.2,
    'Debt_to_Equity': 0.1,
    'ROE': 0.2
}



# Apply score functions
filtered_df_upside.loc[:, 'EPS_CAGR_Score'] = filtered_df_upside['eps_cagr'].apply(score_EPS_CAGR)
filtered_df_upside.loc[:, 'Revenue_CAGR_Score'] = filtered_df_upside['revenue_cagr'].apply(score_Revenue_CAGR)
filtered_df_upside.loc[:, 'ROCE_FY27E_Score'] = filtered_df_upside['roce_fy27e'].apply(score_ROCE_FY27E)
filtered_df_upside.loc[:, 'Debt_to_Equity_Score'] = filtered_df_upside['debt_to_equity'].apply(score_Debt_to_Equity)
filtered_df_upside.loc[:, 'ROE_Score'] = filtered_df_upside['roe_fy27e'].apply(score_ROE)

# Calculate Final Score
filtered_df_upside['Final_Score'] = (
    filtered_df_upside['EPS_CAGR_Score'] * weights_of_parameter['EPS_CAGR'] +
    filtered_df_upside['Revenue_CAGR_Score'] * weights_of_parameter['Revenue_CAGR'] +
    filtered_df_upside['ROCE_FY27E_Score'] * weights_of_parameter['ROCE_FY27E'] +
    filtered_df_upside['Debt_to_Equity_Score'] * weights_of_parameter['Debt_to_Equity'] +
    filtered_df_upside['ROE_Score'] * weights_of_parameter['ROE']
)

# Rank & View Result
filtered_df_upside['Rank'] = filtered_df_upside['Final_Score'].rank(ascending=False)
# print(filtered_df_upside['Rank'].head(10))

df_top = filtered_df_upside.sort_values('Final_Score', ascending=False).reset_index(drop=True)
# print(df_top[['ar_sector','company_name', 'Final_Score']])

df_top['Rank'] = df_top['Final_Score'].rank(method='min', ascending=False).astype(int)

#Select top 30
top_30 = df_top.head(30)

# View top 30 companies with score and rank
# print(top_30[['company_name', 'bb_code','market_cap', 'Final_Score', 'Rank']])

total_market_cap = top_30['market_cap'].sum()
print(total_market_cap)
top_30['Weight'] = top_30['market_cap'] / total_market_cap
# print(top_30[['company_name', 'Weight','upside/downside']])

# Original DataFrame (many columns)
# print("Original columns:", top_30.columns.tolist())

# Cleaned version (keeping only what you need)
required_columns = ['ar_sector','company_name', 'Weight', 'upside/downside']
top_30_clean = top_30[required_columns].copy().reset_index(drop=True)

# print("\nCleaned DataFrame:")
# print(top_30_clean.head())

tickers = [
    "SWSOLAR.NS",
    "DIXON.NS",
    "LLOYDSME.NS",
    "SUZLON.NS",
    "INOXWIND.NS",
    "PAGEIND.NS",
    "BAJAJ-AUTO.NS",
    "TVSMOTOR.NS",
    "CUMMINSIND.NS",
    "PERSISTENT.NS",
    "ABB.NS",
    "MAXHEALTH.NS",
    "DHANUKA.NS",
    "VENUSPIPES.NS",
    "ASHOKLEY.NS",
    "KPITTECH.NS",
    "EMAMILTD.NS",
    "METROBRAND.NS",
    "ARVINDFASN.NS",
    "SUMICHEM.NS",
    "MOIL.NS",
    "LTTS.NS",
    "INDRAMEDCO.NS",
    "UBL.NS",
    "SHARDACROP.NS",
    "ORIENTELEC.NS",
    "BLUESTARCO.NS",
    "PGEL.NS",
    "SYMPHONY.NS",
    "VIPIND.NS"
]
top_30_clean['Ticker'] = tickers  # Assuming 'tickers' is a list of 30 tickers in correct order

# print(len(tickers), len(top_30_clean))
top_30_clean.head()

start_date = "2015-01-01"
end_date = date.today().strftime("%Y-%m-%d")
df_close_price = yf.download(tickers, start=start_date, end=end_date)['Close']
df_close_price.head()

# Step 3: Convert column headers to datetime
df_close_price.index = pd.to_datetime(df_close_price.index)

print("\nStep 3: Datetime Index\n", df_close_price.index)

# Step 4: Create complete calendar range from min to max date
full_range = pd.date_range(start=df_close_price.index.min(), end=df_close_price.index.max(), freq='D')

# Step 5: Reindex to include missing dates and forward fill
df_filled = df_close_price.reindex(full_range).ffill()

# ✅ Step 4: Remove 29th February
df_filled = df_filled[~((df_filled.index.month == 2) & (df_filled.index.day == 29))]
# print("\nStep 4: After removing 29-Feb\n", df_filled.head(10))
# Step 6: Transpose
df_final = df_filled.T
# print("\nStep 6: Transposed \n", df_final.iloc[:, :5])  # Show first 5 dates
df_final.columns = df_final.columns.strftime('%d-%b-%Y')  # Format dates nicely
df_final.reset_index(inplace=True)
df_final.rename(columns={'index': 'Row Labels'}, inplace=True)
# print("\nStep 7: Final Output\n", df_final.head())

# 1. Set 'Ticker' as the index (if not already)
df_final.set_index('Ticker', inplace=True)

# 2. Convert column headers from strings to datetime
df_final.columns = pd.to_datetime(df_final.columns, format='%d-%b-%Y')

# 3. Transpose to make dates the index (for returns calculation)
df_for_returns = df_final.T  # Now rows=dates, columns=tickers

# 4. Initialize returns DataFrame
calendar_returns = pd.DataFrame(index=df_for_returns.index, columns=df_for_returns.columns)

# 5. Calculate 1-year returns
for current_date in df_for_returns.index:
    past_date = current_date.replace(year=current_date.year - 1)
    if past_date in df_for_returns.index:
        calendar_returns.loc[current_date] = (
            (df_for_returns.loc[current_date] / df_for_returns.loc[past_date] - 1)
        )

# 6. Drop rows where all values are NaN
calendar_returns = calendar_returns.dropna(how='all')

# 7. Transpose back to original format (if needed)
calendar_returns_final = calendar_returns.T  # Rows=tickers, Columns=dates

# 8. Preview
# print("1-Year Rolling Returns (%):")
# print(calendar_returns_final.head())

import pandas as pd

# 1. Find the first date with valid data for EACH STOCK (row-wise operation)
start_dates = calendar_returns_final.apply(
    lambda row: row.first_valid_index(),
    axis=1  # Critical: operate on rows, not columns
).to_frame(name='First Valid Date')

# 2. Convert to datetime (now safe because values are actual date strings)
start_dates['First Valid Date'] = pd.to_datetime(start_dates['First Valid Date'])

# 3. Group stocks by their first valid date
grouped = start_dates.groupby('First Valid Date')

# 4. Create the output DataFrame
group_df = pd.DataFrame({
    'First Valid Date': grouped.groups.keys(),
    'Stocks': [list(stocks) for stocks in grouped.groups.values()]
})

# Sort by date for better readability
group_df = group_df.sort_values('First Valid Date')

print("Stocks Grouped by Their First Valid Date:")
print(group_df)

import numpy as np

# Step 5: Initialize full empty covariance matrix
all_stocks = calendar_returns_final.index.tolist()
cov_matrix = pd.DataFrame(np.nan, index=all_stocks, columns=all_stocks)

# Step 6: Rolling Covariance Calculation
for i, row in group_df.iterrows():
    current_stocks = row['Stocks']
    current_data = calendar_returns_final.loc[current_stocks].dropna(axis=1, how='all')

    # A. Internal covariance for current group
    current_cov = current_data.T.cov()
    cov_matrix.loc[current_stocks, current_stocks] = current_cov

    # B. Cross-covariance with prior groups
    if i > 0:
        prior_stocks = group_df.iloc[:i]['Stocks'].sum()  # Flatten list of previous groups
        prior_data = calendar_returns_final.loc[prior_stocks].dropna(axis=1, how='all')

        # Common dates
        common_dates = current_data.columns.intersection(prior_data.columns)
        if not common_dates.empty:
            combined_data = pd.concat([
                current_data[common_dates],
                prior_data[common_dates]
            ])
            combined_cov = combined_data.T.cov()

            # Fill off-diagonal
            cov_matrix.loc[current_stocks, prior_stocks] = combined_cov.loc[current_stocks, prior_stocks]
            cov_matrix.loc[prior_stocks, current_stocks] = combined_cov.loc[prior_stocks, current_stocks]

# Step 7: Fill any remaining NaNs with 0
cov_matrix = cov_matrix.fillna(0)

# Step 8: Output
# print("✅ Annualized Covariance Matrix (shape:", cov_matrix.shape, ")")
# print(cov_matrix.round(3))

# Set lambda (risk aversion)
lambda_risk = 3.0

# Market weights (should sum to 1)
market_weights = top_30_clean['Weight'].values
market_weights = market_weights / market_weights.sum()

# Ensure order matches covariance matrix
cov_ordered = cov_matrix.loc[top_30_clean['Ticker'], top_30_clean['Ticker']]

# Compute implied returns π
pi = lambda_risk * cov_ordered @ market_weights

# Create Series with ticker names
pi = pd.Series(pi, index=top_30_clean['Ticker'], name='Implied Return')
print("✅ Market-implied Returns (π):\n", pi.round(4))

# Q = investor absolute views in same order as tickers
Q = top_30_clean['upside/downside'].values  # shape: (30,)
print(Q)

# P = identity matrix (each view is about 1 asset)
P = np.eye(len(Q))  # shape: (30, 30)
print(P)

tau = 0.05  # You can tweak this
omega = np.diag(np.diag(tau * P @ cov_ordered.values @ P.T))

tau_sigma_inv = np.linalg.inv(tau * cov_ordered.values)
omega_inv = np.linalg.inv(omega)

# Compute posterior mean (BL expected returns)
middle = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
right = tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q

bl_returns = pd.Series(middle @ right, index=top_30_clean['Ticker'], name='BL_Return')

bl_cov = cov_ordered + middle  # Σ + adjustment

sectors = top_30_clean['ar_sector'].unique()
# print(sectors)

sector_max_limits = {
    'Renewable Energy': 0.05,
    'Consumer Durables': 0.05,
    'Metal': 0.05,
    'Retail': 0.05,
    'AUTOMOBILES': 0.10,
    'Industrials': 0.5,
    'IT/ITES/BPM': 0.10,
    'Healthcare': 0.5,
    'Agrochemicals': 0.10,
    'FMCG': 0.10,
    'Alchoholic Beverages': 0.05,
    'Chemical': 0.10,
    'FMEG': 0.10,
    'Luggage': 0.05
}

sector_min_limits = {
    'Renewable Energy': 0.0,
    'Consumer Durables': 0.0,
    'IT/ITES/BPM': 0.0,
    'AUTOMOBILES': 0.0,
    'Healthcare': 0.0,
    'FMCG': 0.0,
    'Alchoholic Beverages': 0.0,
    'Chemical': 0.0,
    'FMEG': 0.0,
    'Luggage': 0.0,
}

sector_matrix = pd.DataFrame(0, index=sector_max_limits.keys(), columns=top_30_clean['Ticker'])

for _, row in top_30_clean.iterrows():
    sector = row['ar_sector']
    ticker = row['Ticker']
    if sector in sector_max_limits:
        sector_matrix.loc[sector, ticker] = 1


def MaximizeSharpeRatioOptmzn(MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize,
                               sector_matrix, sector_max_limits, sector_min_limits):

    # Objective: Maximize Sharpe Ratio (minimize negative Sharpe)
     def sharpe_objective(x, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
        func_denom = np.sqrt(np.matmul(np.matmul(x, CovarReturns), x.T))
        func_numer = np.matmul(np.array(MeanReturns), x.T) - RiskFreeRate
        return -(func_numer / func_denom)

    # Constraint: Fully invested portfolio (sum = 1)
     def fully_invested_constraint(x):
        return np.sum(x) - 1

    # Sector Constraints
     sector_constraints = []

     # Max constraints
     for sector, max_limit in sector_max_limits.items():
        vector = sector_matrix.loc[sector].values
        sector_constraints.append(LinearConstraint(vector, lb=-np.inf, ub=max_limit))

    # Min constraints
     for sector, min_limit in sector_min_limits.items():
        if sector in sector_matrix.index:
            vector = sector_matrix.loc[sector].values
            sector_constraints.append(LinearConstraint(vector, lb=min_limit, ub=np.inf))

    # Initial weights and bounds
     x_init = np.repeat(1 / PortfolioSize, PortfolioSize)
     bounds = tuple((0, 1) for _ in range(PortfolioSize))
     constraints = [{'type': 'eq', 'fun': fully_invested_constraint}] + sector_constraints


    # Run optimizer
     opt = optimize.minimize(
        sharpe_objective,
        x0=x_init,
        args=(MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-3
     )
     return opt

opt_result = MaximizeSharpeRatioOptmzn(
    MeanReturns = bl_returns.values,
    CovarReturns = bl_cov.values,
    RiskFreeRate = 0.05,
    PortfolioSize = len(bl_returns),
    sector_matrix = sector_matrix,
    sector_max_limits = sector_max_limits,
    sector_min_limits = sector_min_limits
)

if opt_result.success:
    top_30_clean['Optimized Weight'] = opt_result.x.round(4)
    result_df = top_30_clean[['Ticker', 'ar_sector', 'Optimized Weight']]
    print(result_df.sort_values('Optimized Weight', ascending=False))
else:
    print(" Optimization failed:", opt_result.message)

weights = opt_result.x
expected_returns = (bl_returns/100).values
cov_matrix = bl_cov.values
risk_free_rate = 0.05

# Expected return
portfolio_return = np.dot(weights, expected_returns)

# Volatility (risk)
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Sharpe Ratio
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# print(f" Portfolio Expected Return: {portfolio_return:.2%}")
# print(f" Portfolio Volatility     : {portfolio_volatility:.2%}")
# print(f" Sharpe Ratio              : {sharpe_ratio:.3f}")

# Ensure prices are clean and in correct ticker order
df_prices = df_filled[top_30_clean['Ticker']].dropna(how='any', axis=0)

# Check that order matches weights
tickers = top_30_clean['Ticker'].tolist()
weights = opt_result.x

# Multiply each stock’s daily price by its weight
df_weighted_price = df_prices.mul(weights, axis=1)

# Sum across stocks to get total portfolio value each day
nav_unscaled = df_weighted_price.sum(axis=1)  # Series: index=date, value=NAV

# Scale such that last value = 100
nav_scaled = nav_unscaled / nav_unscaled.iloc[-1] * 100

plt.figure(figsize=(12, 6))
plt.plot(nav_scaled, color='darkgreen', label='Portfolio NAV (₹100 base)')
plt.title(" Portfolio NAV Based on Weighted Price")
plt.xlabel("Date")
plt.ylabel("NAV (₹)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Get start and end values
nav_start = nav_scaled.iloc[0]
nav_end = nav_scaled.iloc[-1]

# Time duration in years
days = (nav_scaled.index[-1] - nav_scaled.index[0]).days
years = days / 365.25

# CAGR Calculation
cagr = (nav_end / nav_start) ** (1 / years) - 1
print(f" CAGR over {years:.2f} years: {cagr:.2%}")

# Calculate running peak
running_max = nav_scaled.cummax()

# Calculate drawdowns
drawdown = (nav_scaled - running_max) / running_max

# Find max drawdown
max_drawdown = drawdown.min()
drawdown_date = drawdown.idxmin()

print(f" Maximum Drawdown: {max_drawdown:.2%} on {drawdown_date.date()}")

plt.figure(figsize=(12, 4))
plt.plot(drawdown, color='crimson')
plt.title(" Portfolio Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.grid(True)
plt.tight_layout()
plt.show()


nifty200 = yf.download("^CNX200", start=nav_scaled.index[0].strftime("%Y-%m-%d"), end=nav_scaled.index[-1].strftime("%Y-%m-%d"))
nifty200_prices = nifty200['Close'].dropna()

# Align both NAVs to same date range
common_index = nav_scaled.index.intersection(nifty200_prices.index)

# Trim both
nav_aligned = nav_scaled.loc[common_index]
nifty200_aligned = nifty200_prices.loc[common_index]

# Scale NIFTY 200 to end at ₹100
nifty200_nav = nifty200_aligned / nifty200_aligned.iloc[-1] * 100

plt.figure(figsize=(12, 6))
plt.plot(nav_aligned, label='Your Portfolio', color='darkblue')
plt.plot(nifty200_nav, label='NIFTY 200', color='orange')
plt.title(" NAV Comparison: Portfolio vs NIFTY 200")
plt.xlabel("Date")
plt.ylabel("NAV (₹)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Portfolio drawdown
your_peak = nav_aligned.cummax()
your_drawdown = (nav_aligned - your_peak) / your_peak

# NIFTY drawdown
nifty_peak = nifty200_nav.cummax()
nifty_drawdown = (nifty200_nav - nifty_peak) / nifty_peak

plt.figure(figsize=(12, 5))
plt.plot(your_drawdown, label='Your Portfolio', color='crimson')
plt.plot(nifty_drawdown, label='NIFTY 200', color='gray')
plt.title("📉 Drawdown Comparison: Portfolio vs NIFTY 200")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Your Portfolio Drawdown
your_dd_min = your_drawdown.min()
your_dd_date = your_drawdown.idxmin()

# NIFTY Drawdown
nifty_dd_min = nifty_drawdown.min().item()
nifty_dd_date = nifty_drawdown.idxmin()

# If it's still a Series (due to tie), force single value
if isinstance(nifty_dd_date, pd.Series):
    nifty_dd_date = nifty_dd_date.iloc[0]

# Now print
print(f"🔻 Your Max Drawdown : {your_dd_min:.2%} on {your_dd_date.date()}")
print(f"🔻 NIFTY Max Drawdown: {nifty_dd_min:.2%} on {nifty_dd_date.date()}")


# Daily Returns from NAV
nav_series = nav_scaled
nav_returns = nav_series.pct_change().dropna()

# CAGR
start_nav = nav_series.iloc[0]
end_nav = nav_series.iloc[-1]
days = (nav_series.index[-1] - nav_series.index[0]).days
years = days / 365.25

cagr = (end_nav / start_nav) ** (1 / years) - 1

# Annualized Volatility
volatility = nav_returns.std() * np.sqrt(252)

# Sharpe Ratio
risk_free_rate = 0.05  # Assumed annual risk-free rate
sharpe_ratio = (cagr - risk_free_rate) / volatility

# Drawdown and Max Drawdown
running_max = nav_series.cummax()
drawdown = (nav_series - running_max) / running_max
max_drawdown = drawdown.min()
max_dd_date = drawdown.idxmin()

# Output Summary
print("Portfolio Performance from NAV")
print(f"Start Date           : {nav_series.index[0].date()}")
print(f"End Date             : {nav_series.index[-1].date()}")
print(f"Duration (years)     : {years:.2f}")
print(f"Start NAV            : ₹{start_nav:.2f}")
print(f"End NAV              : ₹{end_nav:.2f}")
print(f" CAGR              : {cagr:.2%}")
print(f" Volatility        : {volatility:.2%}")
print(f" Sharpe Ratio       : {sharpe_ratio:.3f}")
print(f" Max Drawdown       : {max_drawdown:.2%} on {max_dd_date.date()}")

top_30_clean["equal_weight"] =  np.repeat(1 / len(bl_returns), len(bl_returns))
top_30_clean["tp"] = top_30["tp"]

n_assets = len(bl_returns)

weights_bl    = opt_result.x
weights_equal = np.repeat(1 / n_assets, n_assets)
weights_mcap = top_30_clean["Weight"].values
tp = top_30_clean["tp"]

# Target NAVs from weighted target prices
nav_bl_tp     = tp.mul(weights_bl).sum()
nav_equal_tp  = tp.mul(weights_equal).sum()
nav_mcap_tp   = tp.mul(weights_mcap).sum()

# Current NAVs (latest day in df_prices)
nav_bl_today     = df_prices.mul(weights_bl, axis=1).sum(axis=1).iloc[-1]
nav_equal_today  = df_prices.mul(weights_equal, axis=1).sum(axis=1).iloc[-1]
nav_mcap_today   = df_prices.mul(weights_mcap, axis=1).sum(axis=1).iloc[-1]

# Expected Returns
blm_returns   = (nav_bl_tp / nav_bl_today) - 1
equal_returns = (nav_equal_tp / nav_equal_today) - 1
mcap_returns  = (nav_mcap_tp / nav_mcap_today) - 1

# Print
print(f" BLM Expected Return   : {blm_returns:.2%}")
print(f" Equal Weight Return   : {equal_returns:.2%}")
print(f" MCap Weighted Return  : {mcap_returns:.2%}")

from datetime import timedelta

# Ensure datetime index
nav_series.index = pd.to_datetime(nav_series.index)

# Filter last 2 years
cutoff_date = nav_series.index.max() - pd.DateOffset(years=2)
nav_2y = nav_series[nav_series.index >= cutoff_date]

running_max = nav_2y.cummax()
drawdown = (nav_2y - running_max) / running_max

nav_df = pd.DataFrame({
    "Date": nav_2y.index,
    "NAV": nav_2y.values,
    "Drawdown": drawdown.values
})


