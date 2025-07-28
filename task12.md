I am currently working on an Agent project. After uploading a file named `sample_sales_data.csv`, I issued the instruction:

> Load the sample\_sales\_data.csv file and show me basic information.

However, the result was unsatisfactory in several ways:

1. The system did not properly handle the edge cases specified in `edge_cases_documentation.md`.
2. The task execution involved unnecessary complexity. Specifically, a very simple task was expanded into multiple redundant steps.
3. An `InterpreterError` occurred repeatedly with the message:

   ```
   Cannot assign to name 'final_answer': doing this would erase the existing tool!
   ```
4. The same block of code was executed in multiple steps (at least 9 times), with minimal or no variation, each time attempting to assign `final_answer` again and again, which kept failing.
5. This led to excessive and inefficient input/output token usage, which could easily be optimized.
6. Ultimately, even though the correct output (from `info()` and `describe(include='all')`) was displayed, the looped retries and error logs indicate the system is not intelligently managing state or learning from its previous failures.

What I expected was a single-step output that reads the CSV file, prints the basic DataFrame structure and statistics, and returns a concise summary result. Instead, the looped behavior and the ignored edge cases from the documentation file show that the agent’s reasoning and tool-use orchestration may not be working as intended.

Can you help me analyze:

* Why the agent failed to stop after the first successful execution?
* Why it couldn't handle the `final_answer` assignment correctly (is this a naming conflict with the tool system)?
* How to make the agent smarter in handling such basic multi-modal tasks without redundancy?
* How to ensure external references like `edge_cases_documentation.md` are actually used in tool-use decision making?

Please keep the context and intermediate behavior in mind while debugging.