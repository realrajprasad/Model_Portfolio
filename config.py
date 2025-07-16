"""
Configuration file for Portfolio Optimization System
===================================================

This file contains all configuration parameters for the portfolio optimization system.
Modify these values to customize the optimization process without changing the main code.
"""

# Data file path
DATA_FILE_PATH = '/content/Data_To_analyse.xlsx'

# Stock filtering parameters
FILTER_PARAMS = {
    'min_market_cap': 2,
    'accepted_recommendations': ['Buy', 'Hold'],
    'min_upside': 0.1
}

# Scoring weights for different financial metrics
PARAMETER_WEIGHTS = {
    'EPS_CAGR': 0.3,
    'Revenue_CAGR': 0.2,
    'ROCE_FY27E': 0.2,
    'Debt_to_Equity': 0.1,
    'ROE': 0.2
}

# Stock tickers for top 30 selected companies
TICKERS = [
    "SWSOLAR.NS", "DIXON.NS", "LLOYDSME.NS", "SUZLON.NS", "INOXWIND.NS",
    "PAGEIND.NS", "BAJAJ-AUTO.NS", "TVSMOTOR.NS", "CUMMINSIND.NS", 
    "PERSISTENT.NS", "ABB.NS", "MAXHEALTH.NS", "DHANUKA.NS", "VENUSPIPES.NS",
    "ASHOKLEY.NS", "KPITTECH.NS", "EMAMILTD.NS", "METROBRAND.NS", 
    "ARVINDFASN.NS", "SUMICHEM.NS", "MOIL.NS", "LTTS.NS", "INDRAMEDCO.NS",
    "UBL.NS", "SHARDACROP.NS", "ORIENTELEC.NS", "BLUESTARCO.NS", 
    "PGEL.NS", "SYMPHONY.NS", "VIPIND.NS"
]

# Sector constraints - Maximum allocation limits
SECTOR_MAX_LIMITS = {
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

# Sector constraints - Minimum allocation limits
SECTOR_MIN_LIMITS = {
    'Renewable Energy': 0.0,
    'Consumer Durables': 0.0,
    'IT/ITES/BPM': 0.0,
    'AUTOMOBILES': 0.0,
    'Healthcare': 0.0,
    'FMCG': 0.0,
    'Alchoholic Beverages': 0.0,
    'Chemical': 0.0,
    'FMEG': 0.0,
    'Luggage': 0.0
}

# Market data parameters
MARKET_DATA_PARAMS = {
    'start_date': "2015-01-01",
    'benchmark_ticker': "^CNX200",  # NIFTY 200 index
    'risk_free_rate': 0.05  # 5% annual risk-free rate
}

# Black-Litterman model parameters
BLACK_LITTERMAN_PARAMS = {
    'lambda_risk': 3.0,  # Risk aversion parameter
    'tau': 0.05  # Uncertainty scaling factor
}

# Optimization parameters
OPTIMIZATION_PARAMS = {
    'method': 'SLSQP',  # Sequential Least Squares Programming
    'tolerance': 1e-3,
    'max_iterations': 1000
}

# Scoring thresholds for EPS CAGR
EPS_CAGR_THRESHOLDS = {
    5: 50,  # Score 5 for EPS CAGR >= 50%
    4: 40,  # Score 4 for EPS CAGR >= 40%
    3: 30,  # Score 3 for EPS CAGR >= 30%
    2: 20,  # Score 2 for EPS CAGR >= 20%
    1: 0    # Score 0 for EPS CAGR < 20%
}

# Scoring thresholds for Revenue CAGR
REVENUE_CAGR_THRESHOLDS = {
    5: 50,  # Score 5 for Revenue CAGR >= 50%
    4: 30,  # Score 4 for Revenue CAGR >= 30%
    3: 20,  # Score 3 for Revenue CAGR >= 20%
    2: 10,  # Score 2 for Revenue CAGR >= 10%
    1: 0    # Score 0 for Revenue CAGR < 10%
}

# Scoring thresholds for ROCE FY27E
ROCE_FY27E_THRESHOLDS = {
    5: 30,  # Score 5 for ROCE >= 30%
    4: 25,  # Score 4 for ROCE >= 25%
    3: 20,  # Score 3 for ROCE >= 20%
    2: 15,  # Score 2 for ROCE >= 15%
    1: 0    # Score 0 for ROCE < 15%
}

# Scoring thresholds for Debt to Equity (lower is better)
DEBT_TO_EQUITY_THRESHOLDS = {
    5: 1.0,   # Score 5 for D/E <= 1.0
    4: 1.5,   # Score 4 for D/E <= 1.5
    3: 2.0,   # Score 3 for D/E <= 2.0
    2: 3.0,   # Score 2 for D/E <= 3.0
    1: float('inf')  # Score 0 for D/E > 3.0
}

# Scoring thresholds for ROE
ROE_THRESHOLDS = {
    5: 30,  # Score 5 for ROE >= 30%
    4: 25,  # Score 4 for ROE >= 25%
    3: 20,  # Score 3 for ROE >= 20%
    2: 15,  # Score 2 for ROE >= 15%
    1: 0    # Score 0 for ROE < 15%
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'figure_size': (12, 6),
    'portfolio_color': 'darkblue',
    'benchmark_color': 'orange',
    'drawdown_color': 'crimson',
    'nav_color': 'darkgreen',
    'grid': True,
    'dpi': 100
}

# Performance analysis parameters
PERFORMANCE_PARAMS = {
    'trading_days_per_year': 252,
    'calendar_days_per_year': 365.25,
    'nav_base_value': 100
}

# Data processing parameters
DATA_PROCESSING_PARAMS = {
    'numeric_columns': ['eps_cagr', 'revenue_cagr', 'roce_fy27e', 'debt_to_equity', 'roe_fy27e'],
    'required_columns': ['ar_sector', 'company_name', 'Weight', 'upside/downside'],
    'remove_leap_day': True  # Remove February 29th from price data
}