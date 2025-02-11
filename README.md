### Summary
This Python script defines an `Evaluator` class designed to assess Arabic text based on multiple evaluation criteria, including linguistic standards, translation quality, cultural alignment, and methodology. The evaluation is performed by interacting with an external language model (e.g., Anthropic's Claude) through API calls. The class supports scoring the texts on each of these criteria and computes summary statistics (mean, median, standard deviation, etc.) for the collected scores. It can handle a dataset of questions and answers and can optionally evaluate the translation quality.

### Requirements
- **Python 3.x**
- **Libraries**:
  - `anthropic`: To interact with the Anthropic API.
  - `time`: For implementing retry logic in case of API failure.
  - `numpy`: For calculating statistical measures (mean, median, std dev).
  - `tqdm`: For progress bar display during evaluations.
  
  Install the required libraries via:
  ```bash
  pip install anthropic numpy tqdm
  ```

### Main Functions

1. **`__init__(self, list_qna, api_key)`**:  
   Initializes the evaluator with a list of question-answer pairs (`list_qna`) and the API key to access the Anthropic API. It also sets up the initial scores for various evaluation categories.

2. **`_call_llm(self, prompt, max_retries=5)`**:  
   Sends a request to the Anthropic model using the provided `prompt`. This function includes retry logic in case of API failure (up to `max_retries` retries).

3. **`_eval_ling_stnd(self, text)`**:  
   Evaluates the linguistic accuracy of the given Arabic text according to grammar, morphology, syntax, and orthographic rules, as well as special cases like poetry or dialect.

4. **`_eval_trans(self, text)`**:  
   Evaluates the translation quality of the given text based on accuracy, context awareness, consistency, grammar, style, and special cases (e.g., poetry, mathematical notations, dialects).

5. **`_eval_cultural(self, text)`**:  
   Evaluates the cultural alignment of the given text with the Arabic-speaking world. The evaluation includes cultural relevance, philosophical and ethical basis, and terminological adaptation.

6. **`_eval_methodology(self, text)`**:  
   Assesses the methodological and structural quality of the dataset or text. Criteria include dataset structure, source validation, and data depth.

7. **`evaluate_all(self, translating_eval=False)`**:  
   Iterates through the entire dataset (`list_qna`), evaluates each text for linguistic, cultural, and methodological criteria. If `translating_eval` is `True`, it also evaluates the translation quality.

8. **`get_evaluation_stats(self)`**:  
   Computes and returns the evaluation statistics (count, mean, median, standard deviation, min, max) for each evaluation category.

9. **`print_evaluation_stats(self)`**:  
   Prints the evaluation statistics to the console in a readable format.

10. **`get_scores(self)`**:  
    Returns the scores for each evaluation category (linguistic, translation, cultural, and methodological).

### Example Usage:
First you need to clone the repository
```git clone https://github.com/serrysibaee/EAED.git ```
```python
# Initialize evaluator with list of Q&A and API key
evaluator = Evaluator(list_qna=my_data, api_key="your-api-key")

# Evaluate all texts in the dataset
evaluator.evaluate_all(translating_eval=True)

# Print detailed evaluation statistics
evaluator.print_evaluation_stats()
```

This class is useful for automatically assessing large datasets of Arabic texts based on various qualitative standards.