# Portfolio Optimization System

A sophisticated stock portfolio optimization system that uses modern portfolio theory, Black-Litterman optimization, and sector constraints to create optimal investment portfolios.

## Features

- **Stock Filtering**: Multi-criteria filtering based on financial metrics
- **Scoring System**: Weighted scoring of stocks based on fundamental analysis
- **Black-Litterman Optimization**: Incorporates investor views into portfolio optimization
- **Sector Constraints**: Ensures diversification through sector allocation limits
- **Performance Analysis**: Comprehensive portfolio performance metrics and benchmarking
- **Visualization**: Professional charts for NAV and drawdown analysis

## System Architecture

The system is organized into modular classes for maximum maintainability:

### Core Classes

1. **StockFilterer**: Filters stocks based on financial criteria
2. **StockScorer**: Scores stocks using financial metrics
3. **DataProcessor**: Handles data cleaning and processing
4. **MarketDataProcessor**: Downloads and processes market data
5. **CovarianceCalculator**: Calculates covariance matrices for optimization
6. **BlackLittermanOptimizer**: Implements Black-Litterman model
7. **PortfolioOptimizer**: Performs constrained portfolio optimization
8. **PerformanceAnalyzer**: Analyzes portfolio performance
9. **PortfolioVisualization**: Creates professional visualizations

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the data file `Data_To_analyse.xlsx` in the correct path as specified in `config.py`

## Usage

### Basic Usage

```python
from portfolio_optimizer import main

# Run the complete optimization process
main()
```

### Customization

All parameters can be customized by modifying `config.py`:

```python
# Example: Change filtering parameters
FILTER_PARAMS = {
    'min_market_cap': 5,  # Increase minimum market cap
    'accepted_recommendations': ['Buy'],  # Only buy recommendations
    'min_upside': 0.15  # Increase minimum upside requirement
}

# Example: Adjust scoring weights
PARAMETER_WEIGHTS = {
    'EPS_CAGR': 0.4,  # Increase weight for EPS growth
    'Revenue_CAGR': 0.2,
    'ROCE_FY27E': 0.2,
    'Debt_to_Equity': 0.1,
    'ROE': 0.1
}
```

## Configuration Parameters

### Stock Filtering
- `min_market_cap`: Minimum market capitalization threshold
- `accepted_recommendations`: List of acceptable analyst recommendations
- `min_upside`: Minimum upside potential required

### Scoring Weights
- `EPS_CAGR`: Weight for earnings growth
- `Revenue_CAGR`: Weight for revenue growth
- `ROCE_FY27E`: Weight for return on capital employed
- `Debt_to_Equity`: Weight for leverage ratio
- `ROE`: Weight for return on equity

### Sector Constraints
- `SECTOR_MAX_LIMITS`: Maximum allocation per sector
- `SECTOR_MIN_LIMITS`: Minimum allocation per sector

### Black-Litterman Parameters
- `lambda_risk`: Risk aversion parameter (default: 3.0)
- `tau`: Uncertainty scaling factor (default: 0.05)

## Algorithm Overview

### 1. Stock Selection Process

1. **Data Loading**: Load stock data from Excel file
2. **Filtering**: Apply multiple financial filters
3. **Scoring**: Score stocks on 0-5 scale for each metric
4. **Ranking**: Calculate weighted final scores
5. **Selection**: Select top 30 stocks

### 2. Portfolio Optimization

1. **Market Data**: Download historical price data
2. **Returns Calculation**: Calculate 1-year rolling returns
3. **Covariance Matrix**: Build covariance matrix with grouped approach
4. **Black-Litterman**: Incorporate investor views (upside/downside)
5. **Optimization**: Maximize Sharpe ratio with sector constraints

### 3. Performance Analysis

1. **NAV Calculation**: Calculate portfolio NAV series
2. **Metrics Calculation**: CAGR, volatility, Sharpe ratio, max drawdown
3. **Benchmarking**: Compare with market index (NIFTY 200)
4. **Visualization**: Generate performance charts

## Key Improvements in Refined Version

### Code Organization
- ✅ **Modular Design**: Separated into logical classes
- ✅ **Configuration Management**: All parameters in separate config file
- ✅ **Better Naming**: Clear, descriptive function and variable names
- ✅ **Documentation**: Comprehensive docstrings and comments

### Code Quality
- ✅ **Error Handling**: Robust error handling throughout
- ✅ **Type Safety**: Better data type management
- ✅ **Performance**: Optimized calculations where possible
- ✅ **Maintainability**: Easy to modify and extend

### Features
- ✅ **Flexible Configuration**: Easy parameter customization
- ✅ **Professional Output**: Well-formatted results and charts
- ✅ **Comprehensive Analysis**: Complete performance metrics
- ✅ **Benchmarking**: Automated benchmark comparison

## Output

The system provides:

1. **Stock Selection Results**: Top 30 stocks with scores and weights
2. **Optimization Results**: Optimal portfolio weights
3. **Performance Metrics**: 
   - CAGR (Compound Annual Growth Rate)
   - Volatility (Annualized)
   - Sharpe Ratio
   - Maximum Drawdown
4. **Benchmark Comparison**: Performance vs market index
5. **Visualizations**:
   - Portfolio NAV chart
   - NAV comparison with benchmark
   - Drawdown comparison chart

## Example Output

```
Selected top 30 stocks with total market cap: 1,234,567.89

Optimization successful!
         Ticker           ar_sector  Optimized Weight
0   PERSISTENT.NS        IT/ITES/BPM            0.0850
1        DIXON.NS  Consumer Durables            0.0425
...

PORTFOLIO PERFORMANCE SUMMARY
==================================================
Start Date: 2015-01-01
End Date: 2024-12-20
Duration: 9.97 years
CAGR: 15.23%
Volatility: 18.45%
Sharpe Ratio: 0.553
Max Drawdown: -32.15% on 2020-03-23

Portfolio Max Drawdown: -32.15%
CNX200 Max Drawdown: -38.92%

EXPECTED RETURNS COMPARISON
==================================================
Black-Litterman Expected Return: 18.45%
Equal Weight Expected Return: 16.23%
Market Cap Weighted Return: 14.87%
```

## Dependencies

- pandas>=1.5.0
- numpy>=1.20.0
- yfinance>=0.2.0
- matplotlib>=3.5.0
- scipy>=1.8.0
- openpyxl>=3.0.0

## Technical Notes

### Algorithm Preservation
- All original algorithms have been preserved exactly
- No changes to mathematical formulations
- Same optimization methodology (Black-Litterman + Sector Constraints)

### Performance Optimizations
- Efficient covariance matrix calculation
- Optimized data processing
- Memory-efficient operations

### Error Handling
- Graceful handling of missing data
- Network timeout handling for data downloads
- Validation of input parameters

## Future Enhancements

Potential areas for extension:
- Additional risk models (Factor models, GARCH)
- Alternative optimization objectives (Risk parity, Minimum variance)
- Real-time data integration
- Portfolio rebalancing strategies
- ESG scoring integration

## License

This is a portfolio optimization system for educational and research purposes.