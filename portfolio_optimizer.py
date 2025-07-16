"""
Stock Portfolio Optimization System
===================================

A comprehensive portfolio optimization system that:
1. Filters stocks based on financial metrics
2. Scores and ranks companies
3. Uses Black-Litterman optimization with sector constraints
4. Performs portfolio performance analysis and benchmarking

Author: Portfolio Optimization System
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
from scipy import optimize
from scipy.optimize import minimize, LinearConstraint
import openpyxl
import warnings

warnings.filterwarnings('ignore')


class StockFilterer:
    """Class for filtering stocks based on various financial criteria."""
    
    @staticmethod
    def filter_by_market_cap(df, min_cap):
        """Filter stocks by minimum market capitalization."""
        filtered_df = df[df['market_cap'] >= min_cap]
        print(f"{len(filtered_df)} stocks passed Market Cap >= {min_cap}")
        return filtered_df
    
    @staticmethod
    def filter_by_recommendation(df, accepted=['Buy', 'Hold']):
        """Filter stocks by analyst recommendations."""
        filtered_df = df[df['recommendation'].isin(accepted)]
        print(f"{len(filtered_df)} stocks passed Recommendation ({' or '.join(accepted)})")
        return filtered_df
    
    @staticmethod
    def filter_by_upside_downside(df, min_upside):
        """Filter stocks by minimum upside potential."""
        filtered_df = df[df['upside/downside'] >= min_upside]
        print(f"{len(filtered_df)} stocks passed Upside >= {min_upside}")
        return filtered_df
    
    @staticmethod
    def filter_by_revenue_cagr(df, min_revenue_cagr):
        """Filter stocks by minimum revenue CAGR."""
        filtered_df = df[df['revenue_cagr'] >= min_revenue_cagr]
        print(f"{len(filtered_df)} stocks passed Revenue CAGR >= {min_revenue_cagr}")
        return filtered_df
    
    @staticmethod
    def filter_by_roce_fy27e(df, min_roce):
        """Filter stocks by minimum ROCE FY27E."""
        filtered_df = df[df['roce_fy27e'] >= min_roce]
        print(f"{len(filtered_df)} stocks passed ROCE >= {min_roce}")
        return filtered_df
    
    @staticmethod
    def filter_by_roe(df, min_roe):
        """Filter stocks by minimum ROE."""
        filtered_df = df[df['roe_fy27e'] >= min_roe]
        print(f"{len(filtered_df)} stocks passed ROE >= {min_roe}")
        return filtered_df
    
    @staticmethod
    def filter_by_debt_to_equity(df, min_debt_to_equity):
        """Filter stocks by minimum debt to equity ratio."""
        filtered_df = df[df['debt_to_equity'] >= min_debt_to_equity]
        print(f"{len(filtered_df)} stocks passed Debt to Equity >= {min_debt_to_equity}")
        return filtered_df
    
    @staticmethod
    def filter_by_eps_cagr(df, min_eps_cagr):
        """Filter stocks by minimum EPS CAGR."""
        filtered_df = df[df['eps_cagr'] >= min_eps_cagr]
        print(f"{len(filtered_df)} stocks passed EPS CAGR >= {min_eps_cagr}")
        return filtered_df


class StockScorer:
    """Class for scoring stocks based on various financial metrics."""
    
    @staticmethod
    def score_eps_cagr(eps_cagr):
        """Score EPS CAGR on a scale of 0-5."""
        if eps_cagr >= 50: return 5
        elif eps_cagr >= 40: return 4
        elif eps_cagr >= 30: return 3
        elif eps_cagr >= 20: return 2
        else: return 0
    
    @staticmethod
    def score_revenue_cagr(revenue_cagr):
        """Score Revenue CAGR on a scale of 0-5."""
        if revenue_cagr >= 50: return 5
        elif revenue_cagr >= 30: return 4
        elif revenue_cagr >= 20: return 3
        elif revenue_cagr >= 10: return 2
        else: return 0
    
    @staticmethod
    def score_roce_fy27e(roce_fy27e):
        """Score ROCE FY27E on a scale of 0-5."""
        if roce_fy27e >= 30: return 5
        elif roce_fy27e >= 25: return 4
        elif roce_fy27e >= 20: return 3
        elif roce_fy27e >= 15: return 2
        else: return 0
    
    @staticmethod
    def score_debt_to_equity(debt_to_equity):
        """Score Debt to Equity ratio on a scale of 0-5 (lower is better)."""
        if debt_to_equity <= 1: return 5
        elif debt_to_equity <= 1.5: return 4
        elif debt_to_equity <= 2: return 3
        elif debt_to_equity <= 3: return 2
        else: return 0
    
    @staticmethod
    def score_roe(roe):
        """Score ROE on a scale of 0-5."""
        if roe >= 30: return 5
        elif roe >= 25: return 4
        elif roe >= 20: return 3
        elif roe >= 15: return 2
        else: return 0


class DataProcessor:
    """Class for processing and cleaning stock data."""
    
    @staticmethod
    def clean_dataframe(df):
        """Clean DataFrame column names for consistency."""
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    
    @staticmethod
    def convert_to_numeric(df, columns):
        """Convert specified columns to numeric type."""
        df_copy = df.copy()
        for col in columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        return df_copy
    
    @staticmethod
    def calculate_final_score(df, weights):
        """Calculate final weighted score for stocks."""
        return (
            df['EPS_CAGR_Score'] * weights['EPS_CAGR'] +
            df['Revenue_CAGR_Score'] * weights['Revenue_CAGR'] +
            df['ROCE_FY27E_Score'] * weights['ROCE_FY27E'] +
            df['Debt_to_Equity_Score'] * weights['Debt_to_Equity'] +
            df['ROE_Score'] * weights['ROE']
        )


class MarketDataProcessor:
    """Class for processing market data from Yahoo Finance."""
    
    @staticmethod
    def download_stock_data(tickers, start_date, end_date):
        """Download stock price data from Yahoo Finance."""
        return yf.download(tickers, start=start_date, end=end_date)['Close']
    
    @staticmethod
    def process_price_data(df_price):
        """Process price data to handle missing dates and format properly."""
        # Convert to datetime
        df_price.index = pd.to_datetime(df_price.index)
        
        # Create complete calendar range
        full_range = pd.date_range(
            start=df_price.index.min(), 
            end=df_price.index.max(), 
            freq='D'
        )
        
        # Reindex and forward fill
        df_filled = df_price.reindex(full_range).ffill()
        
        # Remove February 29th
        df_filled = df_filled[~((df_filled.index.month == 2) & (df_filled.index.day == 29))]
        
        return df_filled
    
    @staticmethod
    def calculate_calendar_returns(df_prices):
        """Calculate 1-year rolling returns for each stock."""
        # Transpose for returns calculation
        df_for_returns = df_prices.T
        
        # Initialize returns DataFrame
        calendar_returns = pd.DataFrame(
            index=df_for_returns.index, 
            columns=df_for_returns.columns
        )
        
        # Calculate 1-year returns
        for current_date in df_for_returns.index:
            past_date = current_date.replace(year=current_date.year - 1)
            if past_date in df_for_returns.index:
                calendar_returns.loc[current_date] = (
                    df_for_returns.loc[current_date] / df_for_returns.loc[past_date] - 1
                )
        
        # Clean and transpose back
        calendar_returns = calendar_returns.dropna(how='all').T
        return calendar_returns


class CovarianceCalculator:
    """Class for calculating covariance matrices for portfolio optimization."""
    
    @staticmethod
    def group_stocks_by_valid_dates(calendar_returns):
        """Group stocks by their first valid date for covariance calculation."""
        start_dates = calendar_returns.apply(
            lambda row: row.first_valid_index(), axis=1
        ).to_frame(name='First Valid Date')
        
        start_dates['First Valid Date'] = pd.to_datetime(start_dates['First Valid Date'])
        grouped = start_dates.groupby('First Valid Date')
        
        group_df = pd.DataFrame({
            'First Valid Date': grouped.groups.keys(),
            'Stocks': [list(stocks) for stocks in grouped.groups.values()]
        })
        
        return group_df.sort_values('First Valid Date')
    
    @staticmethod
    def calculate_covariance_matrix(calendar_returns, group_df):
        """Calculate full covariance matrix using grouped approach."""
        all_stocks = calendar_returns.index.tolist()
        cov_matrix = pd.DataFrame(np.nan, index=all_stocks, columns=all_stocks)
        
        for i, row in group_df.iterrows():
            current_stocks = row['Stocks']
            current_data = calendar_returns.loc[current_stocks].dropna(axis=1, how='all')
            
            # Internal covariance for current group
            current_cov = current_data.T.cov()
            cov_matrix.loc[current_stocks, current_stocks] = current_cov
            
            # Cross-covariance with prior groups
            if i > 0:
                prior_stocks = group_df.iloc[:i]['Stocks'].sum()
                prior_data = calendar_returns.loc[prior_stocks].dropna(axis=1, how='all')
                
                common_dates = current_data.columns.intersection(prior_data.columns)
                if not common_dates.empty:
                    combined_data = pd.concat([
                        current_data[common_dates],
                        prior_data[common_dates]
                    ])
                    combined_cov = combined_data.T.cov()
                    
                    # Fill off-diagonal elements
                    cov_matrix.loc[current_stocks, prior_stocks] = combined_cov.loc[current_stocks, prior_stocks]
                    cov_matrix.loc[prior_stocks, current_stocks] = combined_cov.loc[prior_stocks, current_stocks]
        
        return cov_matrix.fillna(0)


class BlackLittermanOptimizer:
    """Class implementing Black-Litterman optimization model."""
    
    @staticmethod
    def calculate_implied_returns(cov_matrix, market_weights, lambda_risk=3.0):
        """Calculate market-implied returns using reverse optimization."""
        return lambda_risk * cov_matrix @ market_weights
    
    @staticmethod
    def apply_black_litterman(implied_returns, cov_matrix, investor_views, 
                            tau=0.05):
        """Apply Black-Litterman model to incorporate investor views."""
        Q = investor_views
        P = np.eye(len(Q))  # Identity matrix for absolute views
        
        # Calculate uncertainty matrix
        omega = np.diag(np.diag(tau * P @ cov_matrix @ P.T))
        
        # Calculate posterior estimates
        tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
        omega_inv = np.linalg.inv(omega)
        
        middle = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        right = tau_sigma_inv @ implied_returns + P.T @ omega_inv @ Q
        
        bl_returns = middle @ right
        bl_cov = cov_matrix + middle
        
        return bl_returns, bl_cov


class PortfolioOptimizer:
    """Class for portfolio optimization with sector constraints."""
    
    @staticmethod
    def create_sector_matrix(tickers, sectors, sector_limits):
        """Create sector constraint matrix."""
        sector_matrix = pd.DataFrame(0, index=sector_limits.keys(), columns=tickers)
        
        for ticker, sector in zip(tickers, sectors):
            if sector in sector_limits:
                sector_matrix.loc[sector, ticker] = 1
        
        return sector_matrix
    
    @staticmethod
    def maximize_sharpe_ratio(mean_returns, cov_returns, risk_free_rate, 
                            portfolio_size, sector_matrix, sector_max_limits, 
                            sector_min_limits):
        """Optimize portfolio to maximize Sharpe ratio with sector constraints."""
        
        def sharpe_objective(x, mean_returns, cov_returns, risk_free_rate, portfolio_size):
            """Objective function: minimize negative Sharpe ratio."""
            portfolio_return = np.dot(mean_returns, x)
            portfolio_risk = np.sqrt(np.dot(x.T, np.dot(cov_returns, x)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
            return -sharpe
        
        def fully_invested_constraint(x):
            """Constraint: portfolio weights sum to 1."""
            return np.sum(x) - 1
        
        # Sector constraints
        sector_constraints = []
        
        # Maximum sector limits
        for sector, max_limit in sector_max_limits.items():
            if sector in sector_matrix.index:
                vector = sector_matrix.loc[sector].values
                sector_constraints.append(
                    LinearConstraint(vector, lb=-np.inf, ub=max_limit)
                )
        
        # Minimum sector limits
        for sector, min_limit in sector_min_limits.items():
            if sector in sector_matrix.index:
                vector = sector_matrix.loc[sector].values
                sector_constraints.append(
                    LinearConstraint(vector, lb=min_limit, ub=np.inf)
                )
        
        # Setup optimization
        x_init = np.repeat(1 / portfolio_size, portfolio_size)
        bounds = tuple((0, 1) for _ in range(portfolio_size))
        constraints = [{'type': 'eq', 'fun': fully_invested_constraint}] + sector_constraints
        
        # Run optimization
        opt_result = optimize.minimize(
            sharpe_objective,
            x0=x_init,
            args=(mean_returns, cov_returns, risk_free_rate, portfolio_size),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=1e-3
        )
        
        return opt_result


class PerformanceAnalyzer:
    """Class for portfolio performance analysis and benchmarking."""
    
    @staticmethod
    def calculate_portfolio_metrics(weights, expected_returns, cov_matrix, 
                                  risk_free_rate=0.05):
        """Calculate portfolio return, volatility, and Sharpe ratio."""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    @staticmethod
    def calculate_nav_series(price_data, weights):
        """Calculate NAV series for portfolio."""
        df_weighted_price = price_data.mul(weights, axis=1)
        nav_unscaled = df_weighted_price.sum(axis=1)
        nav_scaled = nav_unscaled / nav_unscaled.iloc[-1] * 100
        return nav_scaled
    
    @staticmethod
    def calculate_performance_metrics(nav_series):
        """Calculate comprehensive performance metrics from NAV series."""
        # Daily returns
        nav_returns = nav_series.pct_change().dropna()
        
        # Time calculations
        start_nav = nav_series.iloc[0]
        end_nav = nav_series.iloc[-1]
        days = (nav_series.index[-1] - nav_series.index[0]).days
        years = days / 365.25
        
        # CAGR
        cagr = (end_nav / start_nav) ** (1 / years) - 1
        
        # Volatility
        volatility = nav_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.05
        sharpe_ratio = (cagr - risk_free_rate) / volatility
        
        # Drawdown calculations
        running_max = nav_series.cummax()
        drawdown = (nav_series - running_max) / running_max
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        return {
            'start_date': nav_series.index[0].date(),
            'end_date': nav_series.index[-1].date(),
            'duration_years': years,
            'start_nav': start_nav,
            'end_nav': end_nav,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_dd_date': max_dd_date.date()
        }
    
    @staticmethod
    def compare_with_benchmark(portfolio_nav, benchmark_ticker, start_date, end_date):
        """Compare portfolio performance with benchmark."""
        # Download benchmark data
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)
        benchmark_prices = benchmark_data['Close'].dropna()
        
        # Align dates
        common_index = portfolio_nav.index.intersection(benchmark_prices.index)
        nav_aligned = portfolio_nav.loc[common_index]
        benchmark_aligned = benchmark_prices.loc[common_index]
        
        # Scale benchmark to same base
        benchmark_nav = benchmark_aligned / benchmark_aligned.iloc[-1] * 100
        
        # Calculate benchmark drawdown
        benchmark_peak = benchmark_nav.cummax()
        benchmark_drawdown = (benchmark_nav - benchmark_peak) / benchmark_peak
        
        # Portfolio drawdown
        portfolio_peak = nav_aligned.cummax()
        portfolio_drawdown = (nav_aligned - portfolio_peak) / portfolio_peak
        
        return {
            'portfolio_nav': nav_aligned,
            'benchmark_nav': benchmark_nav,
            'portfolio_drawdown': portfolio_drawdown,
            'benchmark_drawdown': benchmark_drawdown,
            'portfolio_max_dd': portfolio_drawdown.min(),
            'benchmark_max_dd': benchmark_drawdown.min()
        }


class PortfolioVisualization:
    """Class for portfolio visualization and plotting."""
    
    @staticmethod
    def plot_nav_comparison(portfolio_nav, benchmark_nav, title="NAV Comparison"):
        """Plot NAV comparison between portfolio and benchmark."""
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_nav, label='Portfolio', color='darkblue')
        plt.plot(benchmark_nav, label='Benchmark', color='orange')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("NAV (₹)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_drawdown_comparison(portfolio_dd, benchmark_dd, title="Drawdown Comparison"):
        """Plot drawdown comparison between portfolio and benchmark."""
        plt.figure(figsize=(12, 5))
        plt.plot(portfolio_dd, label='Portfolio', color='crimson')
        plt.plot(benchmark_dd, label='Benchmark', color='gray')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_single_nav(nav_series, title="Portfolio NAV", color='darkgreen'):
        """Plot single NAV series."""
        plt.figure(figsize=(12, 6))
        plt.plot(nav_series, color=color, label='Portfolio NAV (₹100 base)')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("NAV (₹)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the portfolio optimization process."""
    
    # Import configuration
    from config import (
        TICKERS, PARAMETER_WEIGHTS, SECTOR_MAX_LIMITS, SECTOR_MIN_LIMITS,
        DATA_FILE_PATH, FILTER_PARAMS, MARKET_DATA_PARAMS, 
        BLACK_LITTERMAN_PARAMS, DATA_PROCESSING_PARAMS
    )
    
    # Load and process data
    print("Loading and processing stock data...")
    df = pd.read_excel(DATA_FILE_PATH)
    df = DataProcessor.clean_dataframe(df)
    
    # Apply filters
    print("\nApplying stock filters...")
    filterer = StockFilterer()
    filtered_df = filterer.filter_by_market_cap(df, min_cap=FILTER_PARAMS['min_market_cap'])
    filtered_df = filterer.filter_by_recommendation(filtered_df, FILTER_PARAMS['accepted_recommendations'])
    filtered_df = filterer.filter_by_upside_downside(filtered_df, min_upside=FILTER_PARAMS['min_upside'])
    
    # Convert to numeric and apply scoring
    filtered_df = DataProcessor.convert_to_numeric(filtered_df, DATA_PROCESSING_PARAMS['numeric_columns'])
    
    # Apply scoring
    scorer = StockScorer()
    filtered_df['EPS_CAGR_Score'] = filtered_df['eps_cagr'].apply(scorer.score_eps_cagr)
    filtered_df['Revenue_CAGR_Score'] = filtered_df['revenue_cagr'].apply(scorer.score_revenue_cagr)
    filtered_df['ROCE_FY27E_Score'] = filtered_df['roce_fy27e'].apply(scorer.score_roce_fy27e)
    filtered_df['Debt_to_Equity_Score'] = filtered_df['debt_to_equity'].apply(scorer.score_debt_to_equity)
    filtered_df['ROE_Score'] = filtered_df['roe_fy27e'].apply(scorer.score_roe)
    
    # Calculate final scores and select top 30
    filtered_df['Final_Score'] = DataProcessor.calculate_final_score(filtered_df, PARAMETER_WEIGHTS)
    top_30 = filtered_df.nlargest(30, 'Final_Score').reset_index(drop=True)
    
    # Calculate market cap weights
    total_market_cap = top_30['market_cap'].sum()
    top_30['Weight'] = top_30['market_cap'] / total_market_cap
    
    # Create clean DataFrame with required columns
    top_30_clean = top_30[DATA_PROCESSING_PARAMS['required_columns']].copy().reset_index(drop=True)
    top_30_clean['Ticker'] = TICKERS
    top_30_clean['tp'] = top_30['tp']
    
    print(f"\nSelected top 30 stocks with total market cap: {total_market_cap}")
    
    # Download and process market data
    print("\nDownloading market data...")
    start_date = MARKET_DATA_PARAMS['start_date']
    end_date = date.today().strftime("%Y-%m-%d")
    
    processor = MarketDataProcessor()
    df_close_price = processor.download_stock_data(TICKERS, start_date, end_date)
    df_filled = processor.process_price_data(df_close_price)
    
    # Calculate returns and covariance matrix
    print("Calculating returns and covariance matrix...")
    calendar_returns = processor.calculate_calendar_returns(df_filled)
    
    cov_calculator = CovarianceCalculator()
    group_df = cov_calculator.group_stocks_by_valid_dates(calendar_returns)
    cov_matrix = cov_calculator.calculate_covariance_matrix(calendar_returns, group_df)
    
    # Black-Litterman optimization
    print("Applying Black-Litterman optimization...")
    market_weights = top_30_clean['Weight'].values
    market_weights = market_weights / market_weights.sum()
    
    bl_optimizer = BlackLittermanOptimizer()
    implied_returns = bl_optimizer.calculate_implied_returns(
        cov_matrix.loc[TICKERS, TICKERS].values, market_weights, 
        BLACK_LITTERMAN_PARAMS['lambda_risk']
    )
    
    investor_views = top_30_clean['upside/downside'].values
    bl_returns, bl_cov = bl_optimizer.apply_black_litterman(
        implied_returns, cov_matrix.loc[TICKERS, TICKERS].values, investor_views,
        BLACK_LITTERMAN_PARAMS['tau']
    )
    
    # Portfolio optimization with sector constraints
    print("Optimizing portfolio with sector constraints...")
    optimizer = PortfolioOptimizer()
    sector_matrix = optimizer.create_sector_matrix(
        TICKERS, top_30_clean['ar_sector'], SECTOR_MAX_LIMITS
    )
    
    opt_result = optimizer.maximize_sharpe_ratio(
        bl_returns, bl_cov, MARKET_DATA_PARAMS['risk_free_rate'], len(TICKERS),
        sector_matrix, SECTOR_MAX_LIMITS, SECTOR_MIN_LIMITS
    )
    
    if opt_result.success:
        top_30_clean['Optimized Weight'] = opt_result.x.round(4)
        print("\nOptimization successful!")
        print(top_30_clean[['Ticker', 'ar_sector', 'Optimized Weight']].sort_values(
            'Optimized Weight', ascending=False
        ))
    else:
        print(f"Optimization failed: {opt_result.message}")
        return
    
    # Performance analysis
    print("\nAnalyzing portfolio performance...")
    analyzer = PerformanceAnalyzer()
    
    # Calculate portfolio metrics
    portfolio_return, portfolio_volatility, sharpe_ratio = analyzer.calculate_portfolio_metrics(
        opt_result.x, bl_returns/100, bl_cov
    )
    
    print(f"Portfolio Expected Return: {portfolio_return:.2%}")
    print(f"Portfolio Volatility: {portfolio_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    
    # Calculate NAV series
    df_prices = df_filled[TICKERS].dropna(how='any', axis=0)
    nav_series = analyzer.calculate_nav_series(df_prices, opt_result.x)
    
    # Performance metrics
    metrics = analyzer.calculate_performance_metrics(nav_series)
    
    print("\n" + "="*50)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Start Date: {metrics['start_date']}")
    print(f"End Date: {metrics['end_date']}")
    print(f"Duration: {metrics['duration_years']:.2f} years")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%} on {metrics['max_dd_date']}")
    
    # Benchmark comparison
    print(f"\nComparing with {MARKET_DATA_PARAMS['benchmark_ticker']}...")
    benchmark_comparison = analyzer.compare_with_benchmark(
        nav_series, MARKET_DATA_PARAMS['benchmark_ticker'], 
        nav_series.index[0].strftime("%Y-%m-%d"),
        nav_series.index[-1].strftime("%Y-%m-%d")
    )
    
    benchmark_name = MARKET_DATA_PARAMS['benchmark_ticker'].replace('^', '')
    print(f"Portfolio Max Drawdown: {benchmark_comparison['portfolio_max_dd']:.2%}")
    print(f"{benchmark_name} Max Drawdown: {benchmark_comparison['benchmark_max_dd']:.2%}")
    
    # Expected returns comparison
    print("\n" + "="*50)
    print("EXPECTED RETURNS COMPARISON")
    print("="*50)
    
    n_assets = len(TICKERS)
    weights_equal = np.repeat(1 / n_assets, n_assets)
    tp = top_30_clean['tp']
    
    # Calculate target NAVs
    nav_bl_tp = tp.mul(opt_result.x).sum()
    nav_equal_tp = tp.mul(weights_equal).sum()
    nav_mcap_tp = tp.mul(market_weights).sum()
    
    # Current NAVs
    nav_bl_today = df_prices.mul(opt_result.x, axis=1).sum(axis=1).iloc[-1]
    nav_equal_today = df_prices.mul(weights_equal, axis=1).sum(axis=1).iloc[-1]
    nav_mcap_today = df_prices.mul(market_weights, axis=1).sum(axis=1).iloc[-1]
    
    # Expected returns
    blm_returns = (nav_bl_tp / nav_bl_today) - 1
    equal_returns = (nav_equal_tp / nav_equal_today) - 1
    mcap_returns = (nav_mcap_tp / nav_mcap_today) - 1
    
    print(f"Black-Litterman Expected Return: {blm_returns:.2%}")
    print(f"Equal Weight Expected Return: {equal_returns:.2%}")
    print(f"Market Cap Weighted Return: {mcap_returns:.2%}")
    
    # Visualization
    print("\nGenerating visualizations...")
    visualizer = PortfolioVisualization()
    
    # Plot portfolio NAV
    visualizer.plot_single_nav(nav_series, "Portfolio NAV Performance")
    
    # Plot comparison with benchmark
    visualizer.plot_nav_comparison(
        benchmark_comparison['portfolio_nav'],
        benchmark_comparison['benchmark_nav'],
        f"Portfolio vs {benchmark_name} Comparison"
    )
    
    # Plot drawdown comparison
    visualizer.plot_drawdown_comparison(
        benchmark_comparison['portfolio_drawdown'],
        benchmark_comparison['benchmark_drawdown'],
        f"Drawdown Comparison: Portfolio vs {benchmark_name}"
    )
    
    print("\nPortfolio optimization completed successfully!")


if __name__ == "__main__":
    main()