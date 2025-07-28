# DataSphere Navigator Agent - Comprehensive Test Plan

## Overview
This test plan covers various user interactions, edge cases, and failure scenarios for the DataSphere Navigator agent to ensure robust performance and prevent infinite loops.

## Test Categories

### 1. Basic Data Loading Tests

#### 1.1 File Upload Tests
**Test Case**: Simple CSV Loading  
**Input**: `"Load sample_sales_data.csv and show me basic information"`  
**Expected Behavior**: 
- Load data once
- Display basic info (shape, columns, data types)
- Use `final_answer` immediately
- **Should NOT** reload data repeatedly

**Test Case**: Excel File Loading  
**Input**: `"Upload and analyze sales_data.xlsx"`  
**Expected Behavior**: Auto-detect Excel format, load successfully

**Test Case**: JSON File Loading  
**Input**: `"Load customer_data.json and show structure"`  
**Expected Behavior**: Parse JSON, convert to DataFrame, show structure

#### 1.2 URL Data Loading Tests
**Test Case**: CSV from URL  
**Input**: `"Load data from https://example.com/data.csv"`  
**Expected Behavior**: Download and load remote CSV file

**Test Case**: Invalid URL  
**Input**: `"Load data from https://invalid-url-123.com/data.csv"`  
**Expected Behavior**: Graceful error handling, no infinite retry

### 2. Analysis Workflow Tests

#### 2.1 Complete Analysis Pipeline
**Test Case**: Full Sales Analysis  
**Input**: `"Analyze sales trends, create visualizations, and provide insights"`  
**Expected Workflow**:
1. Load data (if not already loaded)
2. Profile data quality
3. Create appropriate visualizations
4. Calculate statistics
5. Provide final insights with `final_answer`

#### 2.2 Specific Analysis Requests
**Test Case**: Correlation Analysis  
**Input**: `"Calculate correlations between price and quantity"`  
**Expected Behavior**: Use `calculate_statistics` with correlation analysis

**Test Case**: Visualization Request  
**Input**: `"Create a histogram of product prices"`  
**Expected Behavior**: Use `create_visualization` with histogram type

### 3. Edge Cases and Error Handling

#### 3.1 Data Quality Issues
**Test Case**: Missing Data  
**Input**: Load CSV with 50%+ missing values  
**Expected Behavior**: Identify quality issues, suggest handling strategies

**Test Case**: Empty Dataset  
**Input**: Load empty CSV file  
**Expected Behavior**: Graceful error message, no crash

**Test Case**: Malformed Data  
**Input**: Load CSV with inconsistent column structure  
**Expected Behavior**: Error handling with helpful message

#### 3.2 File Access Issues
**Test Case**: File Not Found  
**Input**: `"Load non_existent_file.csv"`  
**Expected Behavior**: Clear error message, no retry loop

**Test Case**: Permission Denied  
**Input**: Load file without read permissions  
**Expected Behavior**: Permission error handling

**Test Case**: Corrupted File  
**Input**: Load partially corrupted CSV  
**Expected Behavior**: Graceful handling, partial load if possible

### 4. Loop Prevention Tests

#### 4.1 Repeated Tool Calls
**Test Case**: Basic Info Request (Primary Concern)  
**Input**: `"Load sample_sales_data.csv and show me basic information"`  
**Critical Test**: Ensure agent uses `final_answer` after first data load  
**Validation**: No repeated `load_data_from_file` calls

**Test Case**: Already Loaded Data  
**Input**: Make multiple requests on same dataset  
**Expected Behavior**: Skip re-loading, use cached data

#### 4.2 Complex Request Parsing
**Test Case**: Ambiguous Request  
**Input**: `"Tell me about the data"`  
**Expected Behavior**: Provide basic information, not endless analysis

### 5. Multi-Step Analysis Tests

#### 5.1 Sequential Operations
**Test Case**: Step-by-Step Analysis  
**Input**: `"Load data, then create a scatter plot of X vs Y, then calculate correlation"`  
**Expected Behavior**: Execute steps sequentially, avoid redundant operations

#### 5.2 Conditional Analysis
**Test Case**: Data-Dependent Workflow  
**Input**: `"If the data has price column, create price distribution chart"`  
**Expected Behavior**: Check column existence, execute conditionally

### 6. Market Data Tests

#### 6.1 Stock Data
**Test Case**: Stock Analysis  
**Input**: `"Get AAPL stock data for last month"`  
**Expected Behavior**: Fetch data, provide analysis, store for further use

#### 6.2 Crypto Data
**Test Case**: Cryptocurrency  
**Input**: `"Analyze BTC-USD performance"`  
**Expected Behavior**: Fetch crypto data, technical analysis

### 7. Report Generation Tests

#### 7.1 Export Functionality
**Test Case**: PDF Report  
**Input**: `"Generate analysis report in PDF format"`  
**Expected Behavior**: Create comprehensive report with findings

#### 7.2 Multiple Formats
**Test Case**: Format Options  
**Input**: Test HTML, Markdown, PDF export options  
**Expected Behavior**: Generate reports in requested formats

## Test Data Files Required

### Sample CSV Files
1. **sample_sales_data.csv** - Clean sales data with columns: Date, Product, Price, Quantity, Region
2. **missing_data.csv** - Dataset with significant missing values
3. **large_dataset.csv** - Large file to test performance (1M+ rows)
4. **malformed_data.csv** - CSV with inconsistent column structure
5. **empty_file.csv** - Empty CSV file

### Sample Excel Files
1. **sales_data.xlsx** - Multi-sheet Excel with sales data
2. **financial_data.xlsx** - Financial metrics data

### Sample JSON Files
1. **customer_data.json** - Customer information in JSON format
2. **nested_data.json** - Complex nested JSON structure

## Critical Test Scenarios (Anti-Loop)

### Priority 1: Basic Information Request
```
User Input: "Load the sample_sales_data.csv file and show me basic information"
Expected Steps:
1. load_data_from_file("uploads/sample_sales_data.csv")
2. final_answer(data_summary)
MUST NOT: Repeat data loading or call additional analysis tools
```

### Priority 2: Already Loaded Data
```
User Input 1: "Load sample_sales_data.csv"
User Input 2: "Show me basic information"
Expected: Skip re-loading, use cached data, provide final_answer
```

### Priority 3: Simple vs Complex Requests
```
Simple: "Show me the data" → Basic info + final_answer
Complex: "Analyze trends and create visualizations" → Multi-step workflow
```

## Success Criteria

### Loop Prevention
- ✅ No repeated tool calls with identical parameters
- ✅ Immediate `final_answer` for basic information requests
- ✅ Proper termination after completing simple tasks
- ✅ Data reuse without reloading

### Functionality
- ✅ All data formats load successfully
- ✅ Error handling works without crashes
- ✅ Complex analysis workflows complete properly
- ✅ Visualizations generate correctly
- ✅ Reports export in multiple formats

### Performance
- ✅ Response time < 30 seconds for basic operations
- ✅ Memory usage stays reasonable with large datasets
- ✅ Graceful handling of resource limitations

## Test Execution Protocol

1. **Setup**: Ensure all test files are in uploads/ directory
2. **Execution**: Run each test case individually
3. **Monitoring**: Watch for infinite loops (terminate after 10 steps)
4. **Validation**: Verify expected outputs and behaviors
5. **Documentation**: Record any failures or unexpected behaviors

## Regression Testing

After any prompt or tool modifications:
1. Re-run Priority 1-3 critical test scenarios
2. Verify basic information requests still work properly
3. Test edge cases that previously caused issues
4. Confirm complex workflows still function

This test plan ensures the DataSphere Navigator agent provides reliable, efficient data analysis without falling into infinite execution loops.