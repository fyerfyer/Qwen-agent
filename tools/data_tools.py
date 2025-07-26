import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LinearRegression
import warnings
import os
from smolagents import tool
import io
from urllib.parse import urlparse
import tempfile
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64

warnings.filterwarnings('ignore')

# Global variable to store the current dataset
current_dataset = None

@tool
def load_data_from_url(url: str, file_format: str = "auto") -> str:
    """Downloads and loads structured data from a web URL.
    Args:
        url: The web URL containing the data file (CSV, JSON, or Excel)
        file_format: The expected file format ('csv', 'json', 'excel', or 'auto' for auto-detection)
    Returns:
        A string containing the loaded data summary and basic information about the dataset
    """
    global current_dataset
    
    try:
        # Download the file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Auto-detect format if needed
        if file_format == "auto":
            if url.lower().endswith('.csv'):
                file_format = 'csv'
            elif url.lower().endswith('.json'):
                file_format = 'json'
            elif url.lower().endswith(('.xlsx', '.xls')):
                file_format = 'excel'
            else:
                # Try to detect from content
                content_type = response.headers.get('content-type', '').lower()
                if 'json' in content_type:
                    file_format = 'json'
                else:
                    file_format = 'csv'  # Default fallback
        
        # Load data based on format
        if file_format == 'csv':
            current_dataset = pd.read_csv(io.StringIO(response.text))
        elif file_format == 'json':
            data = json.loads(response.text)
            if isinstance(data, list):
                current_dataset = pd.DataFrame(data)
            else:
                current_dataset = pd.json_normalize(data)
        elif file_format == 'excel':
            current_dataset = pd.read_excel(io.BytesIO(response.content))
        else:
            return f"Error: Unsupported file format '{file_format}'"
        
        # Generate summary
        summary = f"""Data successfully loaded from {url}
        
Dataset Information:
- Shape: {current_dataset.shape[0]} rows × {current_dataset.shape[1]} columns
- Columns: {list(current_dataset.columns)}
- Data types: {dict(current_dataset.dtypes)}
- Memory usage: {current_dataset.memory_usage(deep=True).sum() / 1024:.2f} KB

First 5 rows preview:
{current_dataset.head().to_string()}

Missing values per column:
{current_dataset.isnull().sum().to_dict()}
"""
        
        return summary
        
    except requests.exceptions.RequestException as e:
        return f"Error downloading data from URL: {str(e)}"
    except Exception as e:
        return f"Error loading data: {str(e)}"


@tool
def analyze_data_profile(data_description: str) -> str:
    """Analyzes the structure and quality of a dataset.
    Args:
        data_description: Description of the loaded dataset to analyze
    Returns:
        A detailed profile including column statistics, missing values, data types, and quality assessment
    """
    global current_dataset
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_from_url first."
    
    try:
        df = current_dataset.copy()
        
        # Basic statistics
        basic_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'missing_cells': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Column analysis
        column_analysis = []
        for col in df.columns:
            col_info = {
                'column': col,
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_values': df[col].nunique(),
                'memory_usage': df[col].memory_usage(deep=True)
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            elif df[col].dtype == 'object':
                col_info.update({
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'frequency_of_most': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                })
            
            column_analysis.append(col_info)
        
        # Data quality assessment
        quality_issues = []
        
        # Check for high missing data
        high_missing_cols = df.columns[df.isnull().sum() / len(df) > 0.5].tolist()
        if high_missing_cols:
            quality_issues.append(f"High missing data (>50%) in columns: {high_missing_cols}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_issues.append(f"Constant/single-value columns: {constant_cols}")
        
        # Generate comprehensive report
        report = f"""
DATA PROFILE ANALYSIS
=====================

BASIC STATISTICS:
- Total Rows: {basic_stats['total_rows']:,}
- Total Columns: {basic_stats['total_columns']}
- Total Cells: {basic_stats['total_cells']:,}
- Missing Cells: {basic_stats['missing_cells']:,}
- Missing Data Percentage: {basic_stats['missing_percentage']:.2f}%

COLUMN DETAILS:
"""
        
        for col_info in column_analysis:
            report += f"\n{col_info['column']} ({col_info['dtype']}):"
            report += f"\n  - Non-null: {col_info['non_null_count']}/{basic_stats['total_rows']} ({(col_info['non_null_count']/basic_stats['total_rows']*100):.1f}%)"
            report += f"\n  - Unique values: {col_info['unique_values']}"
            
            if 'mean' in col_info:
                report += f"\n  - Mean: {col_info['mean']:.4f}, Std: {col_info['std']:.4f}"
                report += f"\n  - Range: [{col_info['min']:.4f}, {col_info['max']:.4f}]"
            elif 'most_frequent' in col_info:
                report += f"\n  - Most frequent: '{col_info['most_frequent']}' ({col_info['frequency_of_most']} times)"
        
        if quality_issues:
            report += f"\n\nDATA QUALITY ISSUES:\n" + "\n".join(f"- {issue}" for issue in quality_issues)
        else:
            report += f"\n\nDATA QUALITY: No major issues detected"
        
        report += f"\n\nRECOMMENDATIONS:"
        if basic_stats['missing_percentage'] > 10:
            report += f"\n- Consider data imputation or removal strategies for missing values"
        if duplicate_count > 0:
            report += f"\n- Review and potentially remove duplicate rows"
        if len([col for col in df.select_dtypes(include=['object']).columns]) > 0:
            report += f"\n- Consider encoding categorical variables for analysis"
        
        return report
        
    except Exception as e:
        return f"Error analyzing data profile: {str(e)}"


@tool
def create_visualization(data_info: str, chart_type: str, columns: str) -> str:
    """Creates data visualizations based on specified parameters.
    Args:
        data_info: Information about the dataset structure and content
        chart_type: Type of visualization ('histogram', 'scatter', 'line', 'bar', 'heatmap', 'box')
        columns: Comma-separated list of column names to include in the visualization
    Returns:
        A description of the generated visualization with key insights and saved file path
    """
    global current_dataset
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_from_url first."
    
    try:
        df = current_dataset.copy()
        column_list = [col.strip() for col in columns.split(',')]
        
        # Validate columns exist
        missing_cols = [col for col in column_list if col not in df.columns]
        if missing_cols:
            return f"Error: Columns not found in dataset: {missing_cols}"
        
        # Create visualization based on type
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn-v0_8')
        
        insights = []
        
        if chart_type == 'histogram':
            if len(column_list) == 1:
                col = column_list[0]
                if df[col].dtype in ['int64', 'float64']:
                    plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    
                    # Generate insights
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    std_val = df[col].std()
                    skewness = df[col].skew()
                    
                    insights.append(f"Mean: {mean_val:.2f}, Median: {median_val:.2f}")
                    insights.append(f"Standard deviation: {std_val:.2f}")
                    if abs(skewness) > 1:
                        skew_direction = "right" if skewness > 0 else "left"
                        insights.append(f"Distribution is heavily skewed to the {skew_direction}")
                else:
                    return f"Error: Cannot create histogram for non-numeric column '{col}'"
            else:
                return "Error: Histogram requires exactly one numeric column"
        
        elif chart_type == 'scatter':
            if len(column_list) >= 2:
                x_col, y_col = column_list[0], column_list[1]
                if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                    plt.scatter(df[x_col], df[y_col], alpha=0.6, color='coral')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'{y_col} vs {x_col}')
                    
                    # Calculate correlation
                    correlation = df[x_col].corr(df[y_col])
                    insights.append(f"Correlation between {x_col} and {y_col}: {correlation:.3f}")
                    
                    if abs(correlation) > 0.7:
                        insights.append("Strong correlation detected")
                    elif abs(correlation) > 0.3:
                        insights.append("Moderate correlation detected")
                    else:
                        insights.append("Weak correlation detected")
                else:
                    return f"Error: Scatter plot requires numeric columns"
            else:
                return "Error: Scatter plot requires at least two columns"
        
        elif chart_type == 'line':
            for col in column_list:
                if df[col].dtype in ['int64', 'float64']:
                    plt.plot(df.index, df[col], label=col, linewidth=2)
            plt.title('Line Plot')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            
            insights.append(f"Showing trends for {len(column_list)} variables over {len(df)} data points")
        
        elif chart_type == 'bar':
            if len(column_list) == 1:
                col = column_list[0]
                if df[col].dtype == 'object' or df[col].nunique() < 20:
                    value_counts = df[col].value_counts().head(15)
                    plt.bar(range(len(value_counts)), value_counts.values, color='lightgreen')
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                    plt.title(f'Distribution of {col}')
                    plt.ylabel('Count')
                    
                    insights.append(f"Top category: '{value_counts.index[0]}' with {value_counts.iloc[0]} occurrences")
                    insights.append(f"Total unique categories: {df[col].nunique()}")
                else:
                    return f"Error: Too many unique values for bar chart in column '{col}'"
            else:
                return "Error: Bar chart requires exactly one column"
        
        elif chart_type == 'heatmap':
            numeric_cols = [col for col in column_list if df[col].dtype in ['int64', 'float64']]
            if len(numeric_cols) >= 2:
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f')
                plt.title('Correlation Heatmap')
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_val = correlation_matrix.iloc[i, j]
                        corr_pairs.append((numeric_cols[i], numeric_cols[j], abs(corr_val)))
                
                strongest_corr = max(corr_pairs, key=lambda x: x[2])
                insights.append(f"Strongest correlation: {strongest_corr[0]} and {strongest_corr[1]} ({strongest_corr[2]:.3f})")
            else:
                return "Error: Heatmap requires at least two numeric columns"
        
        elif chart_type == 'box':
            numeric_cols = [col for col in column_list if df[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                df[numeric_cols].boxplot()
                plt.title('Box Plot')
                plt.xticks(rotation=45)
                
                # Detect outliers
                outlier_counts = {}
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
                    outlier_counts[col] = len(outliers)
                
                total_outliers = sum(outlier_counts.values())
                insights.append(f"Total outliers detected: {total_outliers}")
                if total_outliers > 0:
                    insights.append(f"Outliers per column: {outlier_counts}")
            else:
                return "Error: Box plot requires at least one numeric column"
        
        else:
            return f"Error: Unsupported chart type '{chart_type}'"
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_{chart_type}_{timestamp}.png"
        filepath = os.path.join(os.getcwd(), filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary
        summary = f"""Visualization Created Successfully!

Chart Type: {chart_type.title()}
Columns Analyzed: {', '.join(column_list)}
File Saved: {filepath}

Key Insights:
{chr(10).join('- ' + insight for insight in insights)}

The visualization shows the relationship and patterns in your data. Use this chart to identify trends, outliers, and correlations that can inform your analysis decisions."""

        return summary
        
    except Exception as e:
        return f"Error creating visualization: {str(e)}"


@tool
def calculate_statistics(data_info: str, analysis_type: str, variables: str) -> str:
    """Performs statistical analysis on specified data variables.
    Args:
        data_info: Information about the dataset to analyze
        analysis_type: Type of statistical analysis ('correlation', 'distribution', 'ttest', 'regression')
        variables: Comma-separated list of variables to include in the analysis
    Returns:
        Statistical results with interpretation and significance levels
    """
    global current_dataset
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_from_url first."
    
    try:
        df = current_dataset.copy()
        variable_list = [var.strip() for var in variables.split(',')]
        
        # Validate variables exist
        missing_vars = [var for var in variable_list if var not in df.columns]
        if missing_vars:
            return f"Error: Variables not found in dataset: {missing_vars}"
        
        results = f"STATISTICAL ANALYSIS RESULTS\n{'='*40}\n"
        results += f"Analysis Type: {analysis_type.upper()}\n"
        results += f"Variables: {', '.join(variable_list)}\n\n"
        
        if analysis_type == 'correlation':
            numeric_vars = [var for var in variable_list if df[var].dtype in ['int64', 'float64']]
            if len(numeric_vars) < 2:
                return "Error: Correlation analysis requires at least two numeric variables"
            
            results += "CORRELATION ANALYSIS:\n"
            correlation_matrix = df[numeric_vars].corr()
            results += f"\nCorrelation Matrix:\n{correlation_matrix.round(4)}\n"
            
            # Statistical significance testing
            results += "\nSignificance Testing (p-values):\n"
            for i, var1 in enumerate(numeric_vars):
                for j, var2 in enumerate(numeric_vars):
                    if i < j:  # Avoid duplicates
                        corr_coef, p_value = pearsonr(df[var1].dropna(), df[var2].dropna())
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                        results += f"{var1} vs {var2}: r={corr_coef:.4f}, p={p_value:.4f} {significance}\n"
            
            results += "\nInterpretation:\n"
            results += "*** p<0.001 (highly significant), ** p<0.01 (very significant), * p<0.05 (significant), n.s. (not significant)\n"
            
            # Find strongest correlations
            strong_corrs = []
            for i in range(len(numeric_vars)):
                for j in range(i+1, len(numeric_vars)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_corrs.append((numeric_vars[i], numeric_vars[j], corr_val))
            
            if strong_corrs:
                results += f"\nStrong Correlations (|r| > 0.5):\n"
                for var1, var2, corr in strong_corrs:
                    direction = "positive" if corr > 0 else "negative"
                    results += f"- {var1} and {var2}: {direction} correlation (r={corr:.3f})\n"
        
        elif analysis_type == 'distribution':
            results += "DISTRIBUTION ANALYSIS:\n"
            for var in variable_list:
                if df[var].dtype in ['int64', 'float64']:
                    data = df[var].dropna()
                    results += f"\n{var} Distribution:\n"
                    results += f"  Count: {len(data)}\n"
                    results += f"  Mean: {data.mean():.4f}\n"
                    results += f"  Median: {data.median():.4f}\n"
                    results += f"  Std Dev: {data.std():.4f}\n"
                    results += f"  Min: {data.min():.4f}\n"
                    results += f"  Max: {data.max():.4f}\n"
                    results += f"  Skewness: {data.skew():.4f}\n"
                    results += f"  Kurtosis: {data.kurtosis():.4f}\n"
                    
                    # Normality test
                    if len(data) > 3:
                        try:
                            stat, p_value = stats.normaltest(data)
                            is_normal = p_value > 0.05
                            results += f"  Normality Test: {'Normal' if is_normal else 'Non-normal'} (p={p_value:.4f})\n"
                        except:
                            results += f"  Normality Test: Could not compute\n"
                
                elif df[var].dtype == 'object':
                    value_counts = df[var].value_counts()
                    results += f"\n{var} Categories:\n"
                    results += f"  Unique values: {df[var].nunique()}\n"
                    results += f"  Most frequent: '{value_counts.index[0]}' ({value_counts.iloc[0]} times)\n"
                    results += f"  Distribution:\n"
                    for cat, count in value_counts.head(10).items():
                        percentage = (count / len(df)) * 100
                        results += f"    {cat}: {count} ({percentage:.1f}%)\n"
        
        elif analysis_type == 'ttest':
            if len(variable_list) < 2:
                return "Error: T-test requires at least two variables"
            
            var1, var2 = variable_list[0], variable_list[1]
            
            if df[var1].dtype in ['int64', 'float64'] and df[var2].dtype in ['int64', 'float64']:
                data1 = df[var1].dropna()
                data2 = df[var2].dropna()
                
                # Independent samples t-test
                t_stat, p_value = ttest_ind(data1, data2)
                
                results += "INDEPENDENT T-TEST RESULTS:\n"
                results += f"Variable 1: {var1} (n={len(data1)}, mean={data1.mean():.4f})\n"
                results += f"Variable 2: {var2} (n={len(data2)}, mean={data2.mean():.4f})\n"
                results += f"T-statistic: {t_stat:.4f}\n"
                results += f"P-value: {p_value:.6f}\n"
                
                significance = "highly significant" if p_value < 0.001 else "very significant" if p_value < 0.01 else "significant" if p_value < 0.05 else "not significant"
                results += f"Result: The difference is {significance}\n"
                
                if p_value < 0.05:
                    higher_mean = var1 if data1.mean() > data2.mean() else var2
                    results += f"Conclusion: {higher_mean} has significantly higher values\n"
                else:
                    results += f"Conclusion: No significant difference between groups\n"
            else:
                return "Error: T-test requires numeric variables"
        
        elif analysis_type == 'regression':
            if len(variable_list) < 2:
                return "Error: Regression analysis requires at least two variables (X and Y)"
            
            # Use first variable as dependent (Y) and others as independent (X)
            y_var = variable_list[0]
            x_vars = variable_list[1:]
            
            # Check if all variables are numeric
            all_vars = [y_var] + x_vars
            numeric_vars = [var for var in all_vars if df[var].dtype in ['int64', 'float64']]
            if len(numeric_vars) != len(all_vars):
                return "Error: Regression analysis requires all variables to be numeric"
            
            # Prepare data
            clean_data = df[all_vars].dropna()
            X = clean_data[x_vars]
            y = clean_data[y_var]
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate metrics
            r_squared = model.score(X, y)
            y_pred = model.predict(X)
            
            results += "LINEAR REGRESSION ANALYSIS:\n"
            results += f"Dependent Variable: {y_var}\n"
            results += f"Independent Variables: {', '.join(x_vars)}\n"
            results += f"Sample Size: {len(clean_data)}\n"
            results += f"R-squared: {r_squared:.4f} ({r_squared*100:.1f}% of variance explained)\n"
            results += f"Intercept: {model.intercept_:.4f}\n"
            
            results += "\nCoefficients:\n"
            for i, var in enumerate(x_vars):
                results += f"  {var}: {model.coef_[i]:.4f}\n"
            
            # Model interpretation
            if r_squared > 0.7:
                strength = "strong"
            elif r_squared > 0.3:
                strength = "moderate"
            else:
                strength = "weak"
            
            results += f"\nModel Quality: {strength.title()} predictive power\n"
            
            # Find most important predictors
            coef_importance = [(x_vars[i], abs(model.coef_[i])) for i in range(len(x_vars))]
            coef_importance.sort(key=lambda x: x[1], reverse=True)
            results += f"Most Important Predictor: {coef_importance[0][0]}\n"
        
        else:
            return f"Error: Unsupported analysis type '{analysis_type}'"
        
        return results
        
    except Exception as e:
        return f"Error in statistical analysis: {str(e)}"


@tool
def get_market_data(symbol: str, data_type: str = "stock", period: str = "1d") -> str:
    """Retrieves current market data for financial analysis.
    Args:
        symbol: The ticker symbol or identifier (e.g., 'AAPL', 'BTC-USD')
        data_type: Type of market data ('stock', 'crypto', 'forex', 'commodity')
        period: Time period for data ('1d', '5d', '1mo', '3mo', '1y')
    Returns:
        Current market data including price, volume, and basic technical indicators
    """
    global current_dataset
    
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        hist_data = ticker.history(period=period)
        
        if hist_data.empty:
            return f"Error: No data found for symbol '{symbol}'. Please check the symbol and try again."
        
        # Get current info
        try:
            info = ticker.info
        except:
            info = {}
        
        # Store historical data in current_dataset for further analysis
        current_dataset = hist_data.copy()
        current_dataset.reset_index(inplace=True)
        
        # Calculate technical indicators
        current_price = hist_data['Close'].iloc[-1]
        prev_close = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
        
        # Moving averages
        if len(hist_data) >= 20:
            ma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1]
        else:
            ma_20 = hist_data['Close'].mean()
        
        if len(hist_data) >= 50:
            ma_50 = hist_data['Close'].rolling(window=50).mean().iloc[-1]
        else:
            ma_50 = hist_data['Close'].mean()
        
        # Volatility
        volatility = hist_data['Close'].pct_change().std() * 100
        
        # Volume analysis
        avg_volume = hist_data['Volume'].mean()
        current_volume = hist_data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # High/Low analysis
        period_high = hist_data['High'].max()
        period_low = hist_data['Low'].min()
        
        # Generate report
        report = f"""MARKET DATA ANALYSIS - {symbol.upper()}
{'='*50}

CURRENT PRICE INFORMATION:
- Current Price: ${current_price:.2f}
- Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
- Data Type: {data_type.title()}
- Period: {period}

PRICE LEVELS:
- Period High: ${period_high:.2f}
- Period Low: ${period_low:.2f}
- 20-Day MA: ${ma_20:.2f}
- 50-Day MA: ${ma_50:.2f}

TECHNICAL INDICATORS:
- Volatility: {volatility:.2f}%
- Volume Ratio: {volume_ratio:.2f}x (vs average)
- Average Volume: {avg_volume:,.0f}
- Current Volume: {current_volume:,.0f}

POSITION RELATIVE TO RANGE:
- Distance from High: {((current_price - period_high) / period_high * 100):+.2f}%
- Distance from Low: {((current_price - period_low) / period_low * 100):+.2f}%

"""
        
        # Add company info if available
        if info and 'longName' in info:
            report += f"COMPANY INFO:\n"
            report += f"- Name: {info.get('longName', 'N/A')}\n"
            report += f"- Sector: {info.get('sector', 'N/A')}\n"
            report += f"- Market Cap: ${info.get('marketCap', 0):,.0f}\n" if info.get('marketCap') else ""
        
        # Technical analysis insights
        report += f"\nMARKET INSIGHTS:\n"
        
        if price_change_pct > 2:
            report += f"- Strong upward movement (+{price_change_pct:.1f}%)\n"
        elif price_change_pct < -2:
            report += f"- Strong downward movement ({price_change_pct:.1f}%)\n"
        else:
            report += f"- Moderate price movement ({price_change_pct:+.1f}%)\n"
        
        if current_price > ma_20:
            report += f"- Trading above 20-day moving average (bullish signal)\n"
        else:
            report += f"- Trading below 20-day moving average (bearish signal)\n"
        
        if volume_ratio > 1.5:
            report += f"- High volume activity ({volume_ratio:.1f}x average)\n"
        elif volume_ratio < 0.5:
            report += f"- Low volume activity ({volume_ratio:.1f}x average)\n"
        
        if volatility > 3:
            report += f"- High volatility ({volatility:.1f}%) - increased risk\n"
        elif volatility < 1:
            report += f"- Low volatility ({volatility:.1f}%) - stable movement\n"
        
        report += f"\nDATA SUMMARY:\n"
        report += f"- {len(hist_data)} data points loaded for period {period}\n"
        report += f"- Date range: {hist_data.index[0].strftime('%Y-%m-%d')} to {hist_data.index[-1].strftime('%Y-%m-%d')}\n"
        report += f"- Data stored in current dataset for further analysis\n"
        
        return report
        
    except Exception as e:
        return f"Error retrieving market data for {symbol}: {str(e)}"


@tool
def export_analysis_report(analysis_summary: str, format_type: str = "pdf") -> str:
    """Generates a comprehensive analysis report with findings and visualizations.
    Args:
        analysis_summary: Summary of all analyses performed during the session
        format_type: Output format for the report ('pdf', 'html', 'markdown')
    Returns:
        Path to the generated report file with executive summary
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate comprehensive report content
        report_content = f"""# DataSphere Navigator Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
This report contains a comprehensive analysis of the dataset(s) processed during this session. The analysis includes data profiling, statistical computations, visualizations, and key insights derived from the data.

## Analysis Overview
{analysis_summary}

## Dataset Information
"""
        
        global current_dataset
        if current_dataset is not None:
            report_content += f"""
### Current Dataset Statistics
- Shape: {current_dataset.shape[0]} rows × {current_dataset.shape[1]} columns
- Columns: {', '.join(current_dataset.columns.tolist())}
- Memory Usage: {current_dataset.memory_usage(deep=True).sum() / 1024:.2f} KB

### Data Types
{current_dataset.dtypes.to_string()}

### Missing Data Summary
{current_dataset.isnull().sum().to_string()}

### Basic Statistics (Numeric Columns)
{current_dataset.describe().to_string() if len(current_dataset.select_dtypes(include=['number']).columns) > 0 else 'No numeric columns available'}
"""
        else:
            report_content += "\nNo dataset currently loaded in the system."
        
        # Add recommendations section
        report_content += f"""

## Key Insights and Recommendations

Based on the analysis performed, here are the key findings:

1. **Data Quality**: Review the missing data patterns and consider appropriate handling strategies.
2. **Statistical Patterns**: Look for correlations and relationships that can inform decision-making.
3. **Visualization Insights**: Use the generated charts to communicate findings effectively.
4. **Next Steps**: Consider additional analyses based on the patterns discovered.

## Technical Notes

- All analyses were performed using Python scientific computing libraries
- Statistical significance was tested where applicable
- Visualizations were generated with publication-quality settings
- This report was generated by DataSphere Navigator AI Agent

---
*Report generated automatically by DataSphere Navigator*
"""
        
        # Save based on format type
        if format_type == "markdown":
            filename = f"analysis_report_{timestamp}.md"
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        elif format_type == "html":
            # Convert markdown to HTML (simple conversion)
            processed_content = report_content.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>').replace('\n\n', '<br><br>').replace('\n', '<br>')
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>DataSphere Navigator Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{processed_content}
</body>
</html>"""
            
            filename = f"analysis_report_{timestamp}.html"
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        elif format_type == "pdf":
            # For PDF, we'll create HTML first then note that PDF conversion would require additional libraries
            filename = f"analysis_report_{timestamp}.html"
            filepath = os.path.join(os.getcwd(), filename)
            
            processed_content_pdf = report_content.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>').replace('\n\n', '</p><p>').replace('\n', '<br>')
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>DataSphere Navigator Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .footer {{ text-align: center; margin-top: 30px; font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
<div class="header">
    <h1>DataSphere Navigator Analysis Report</h1>
    <p>Professional Data Analysis Report</p>
</div>
{processed_content_pdf}
<div class="footer">
    <p>Generated by DataSphere Navigator AI Agent</p>
</div>
</body>
</html>"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            note = " (HTML format - use browser 'Print to PDF' for PDF conversion)"
        
        else:
            return f"Error: Unsupported format type '{format_type}'. Supported formats: pdf, html, markdown"
        
        # Generate summary
        summary = f"""Analysis Report Generated Successfully!

File Details:
- Format: {format_type.upper()}
- Filename: {filename}
- Location: {filepath}
- Size: {os.path.getsize(filepath) / 1024:.2f} KB

Report Contents:
- Executive summary of all analyses
- Dataset statistics and quality assessment
- Technical insights and recommendations
- Formatted for professional presentation

The report provides a comprehensive overview of your data analysis session and can be shared with stakeholders or used for documentation purposes.

To access the report, open the file at: {filepath}
"""
        
        return summary
        
    except Exception as e:
        return f"Error generating analysis report: {str(e)}"