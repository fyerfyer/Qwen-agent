I have implemented a simple data agent. When I ran `python app.py` command, I got the following error:

```
╭────────────────────────────────────────────────────── New run - DataSphere_Navigator ──────────────────────────────────────────────────────╮
│                                                                                                                                            │
│ Load the sample_sales_data.csv file and show me basic information                                                                          │
│ You have been provided with these files, which might be helpful or not: ['uploads/sample_sales_data.csv'\]                                 │
│                                                                                                                                            │
╰─ OllamaInferenceClient - qwen2.5:7b ───────────────────────────────────────────────────────────────────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing parsed code: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  import pandas as pd                                                                                                                         
                                                                                                                                              
  # Load the sales data into a DataFrame                                                                                                      
  sales_df = pd.read_csv('uploads/sample_sales_data.csv')                                                                                     
                                                                                                                                              
  # Get basic information about the dataframe                                                                                                 
  basic_info = sales_df.info()                                                                                                                
                                                                                                                                              
  # Display the basic information                                                                                                             
  print(basic_info)                                                                                                                           
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when
initializing your CodeAgent.
Code execution failed at line 'import pandas as pd' due to: InterpreterError: Import of pandas is not allowed. Authorized imports are: 
['random', 'math', 'time', 'unicodedata', 'queue', 'statistics', 'itertools', 're', 'collections', 'stat', 'datetime'\]
[Step 1: Duration 37.87 seconds| Input tokens: 3,491 | Output tokens: 104]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing parsed code: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  import pandas as pd                                                                                                                         
                                                                                                                                              
  # Load the sales data into a DataFrame                                                                                                      
  sales_df = pd.read_csv('uploads/sample_sales_data.csv')                                                                                     
                                                                                                                                              
  # Get basic information about the dataframe                                                                                                 
  basic_info = sales_df.info()                                                                                                                
                                                                                                                                              
  # Display the basic information                                                                                                             
  print(basic_info)                                                                                                                           
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when
initializing your CodeAgent.
Code execution failed at line 'import pandas as pd' due to: InterpreterError: Import of pandas is not allowed. Authorized imports are: 
['random', 'math', 'time', 'unicodedata', 'queue', 'statistics', 'itertools', 're', 'collections', 'stat', 'datetime'\]
[Step 2: Duration 3.67 seconds| Input tokens: 7,103 | Output tokens: 169]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing parsed code: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  import pandas as pd                                                                                                                         
                                                                                                                                              
  # Load the sales data into a DataFrame                                                                                                      
  sales_df = pd.read_csv('uploads/sample_sales_data.csv')                                                                                     
                                                                                                                                              
  # Get basic information about the dataframe                                                                                                 
  basic_info = sales_df.info()                                                                                                                
                                                                                                                                              
  # Display the basic information                                                                                                             
  print(basic_info)                                                                                                                           
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when
initializing your CodeAgent.
Code execution failed at line 'import pandas as pd' due to: InterpreterError: Import of pandas is not allowed. Authorized imports are: 
['random', 'math', 'time', 'unicodedata', 'queue', 'statistics', 'itertools', 're', 'collections', 'stat', 'datetime'\]
[Step 3: Duration 3.67 seconds| Input tokens: 10,795 | Output tokens: 234]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing parsed code: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  import pandas as pd                                                                                                                         
                                                                                                                                              
  # Load the sales data into a DataFrame                                                                                                      
  sales_df = pd.read_csv('uploads/sample_sales_data.csv')                                                                                     
                                                                                                                                              
  # Get basic information about the dataframe                                                                                                 
  basic_info = sales_df.info()                                                                                                                
                                                                                                                                              
  # Display the basic information                                                                                                             
  print(basic_info)                                                                                                                           
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when
initializing your CodeAgent.
Code execution failed at line 'import pandas as pd' due to: InterpreterError: Import of pandas is not allowed. Authorized imports are: 
['random', 'math', 'time', 'unicodedata', 'queue', 'statistics', 'itertools', 're', 'collections', 'stat', 'datetime'\]
[Step 4: Duration 3.70 seconds| Input tokens: 14,567 | Output tokens: 299]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
```

Why does the error occur? Please analyze related code carefully and help me fix the error. 

ote: you only need to check if there are any Python syntax errors in the current code and fix the error yourself. You do **not** need to actually run the code—I will run it myself on the cloud server. 