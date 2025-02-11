import anthropic
import time
import numpy as np
from tqdm import tqdm

class Evaluator:
    def __init__(self, list_qna, api_key):
        self.dataset = list_qna
        self.api_key = api_key
        self.client_claude = anthropic.Anthropic(api_key=self.api_key)
        self.scores = {
            "linguistic": [],
            "translation": [],
            "cultural": [],
            "methodological": []
        }

    def _call_llm(self, prompt, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                response = self.client_claude.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=100,
                    system="أنت مساعد خبير في تقييم النصوص العربية وفقًا لمعايير محددة.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return int(response.content[0].text.strip())  # Convert response to integer score
            except Exception as e:
                retries += 1
                print(f"ERROR: {e}")
                print(f"🔄 Retrying in 3 seconds...")
                time.sleep(3)
        return None  # Return None if all retries fail

    def _eval_ling_stnd(self, text):
        prompt = f"""
        You are an expert Arabic linguist and evaluator. Your task is to assess a given Arabic text based on strict linguistic standards.

        Evaluation Criteria:
        1. Linguistic Accuracy (6 points)
           - Does the text follow Arabic grammar, morphology, syntax, and orthographic rules?
           - Does it avoid weak linguistic structures, even if grammatically correct?
           - Does it use appropriate stylistic choices for the intended purpose?

        2. Special Cases (4 points)
           - If the text contains poetry, is its structure and prosody preserved?
           - Are mathematical notations formatted correctly according to Arabic conventions?
           - If dialectal Arabic is used, does it follow a consistent and standardized framework?

        Task:
        
        - Assign a final score between 1 and 10 based on the text’s quality.

        Text:
        {text}

        Output Format (print only a number):
        X
        """
        score = self._call_llm(prompt)
        if score is not None:
            self.scores["linguistic"].append(score)

    def _eval_trans(self, text):
        prompt = f"""
        You are a highly skilled Arabic translator. Your task is to evaluate the given translation while adhering to strict linguistic and stylistic standards.

        Translation Guidelines:
        - Accuracy: Translate all terms precisely. If a term should remain in its original language, transliterate it and provide the original in brackets.
        - Context Awareness: Avoid literal translations. Adapt the meaning naturally while maintaining clarity and coherence.
        - Consistency: Ensure that the same term is translated consistently throughout the text.
        - Grammar & Style: Follow proper Arabic grammar, morphology, and syntax. Avoid unnatural or weak linguistic structures, even if grammatically correct.
        - Special Cases:
          - For poetry, maintain the structure and prosody.
          - For mathematical notations, follow Arabic conventions or provide clear rules for using Latin symbols.
          - For dialects, adhere to a standardized framework for representation.

        Task:
        Evaluate the given translation according to previous guidelines.

        Text:
        {text}

        Output Format (print only a number):
        X
        """
        score = self._call_llm(prompt)
        if score is not None:
            self.scores["translation"].append(score)

    def _eval_cultural(self, text):
        prompt = f"""
        You are an expert in Arabic language and culture. Your task is to evaluate a given text based on its cultural alignment with the Arabic-speaking world.

        Evaluation Criteria:
        1. Cultural Relevance (4 points)
           - Does the text align with the historical, social, and cultural contexts of the Arabic-speaking world?
           - Does it avoid irrelevant or Western-specific references that do not fit Arab culture?

        2. Philosophical and Ethical Basis (3 points)
           - Does the text refrain from presenting Western philosophical or ethical concepts as universal truths without explanation or adaptation?
           - Does it avoid expressions or examples that conflict with Arab cultural norms?

        3. Terminological Adaptation (3 points)
           - Are Westernized terms replaced with culturally and linguistically appropriate Arabic equivalents?
           - Are Arabic equivalents or transliterations provided where necessary, maintaining cultural integrity?

        Task:
        
        - Assign a final score between 1 and 10 based on cultural alignment.

        Text:
        {text}

        Output Format (print only a number):
        X
        """
        score = self._call_llm(prompt)
        if score is not None:
            self.scores["cultural"].append(score)

    def _eval_methodology(self, text):
        prompt = f"""
        You are an expert in Arabic data structuring and methodology. Your task is to evaluate a given dataset or text based on its adherence to methodological and structural standards.

        Evaluation Criteria:
        1. Dataset Structure (3 points)
           - Are questions and content logically organized within their relevant categories?
           - Is redundancy or confusion avoided by grouping related queries appropriately?
           - Is the information up-to-date and includes accurate dates where applicable?

        2. Source Validation (4 points)
           - Are knowledge and data attributed to original Arabic primary sources, such as books, studies, and statistical reports relevant to Arabic societies?
           - Is there minimal reliance on non-Arabic secondary references when constructing Arabic datasets?
           - Are Quranic texts written with complete accuracy using the Uthmanic script?

        3. Data Depth (3 points)
           - Does the dataset reflect depth and richness, avoiding overly simplistic or shallow questions and answers?
           - Does it incorporate diverse perspectives within the Arabic-speaking world for inclusivity?

        Task:
        
        - Assign a final score between 1 and 10 based on methodological and structural quality.

        Text:
        {text}

        Output Format (print only a number):
        X
        """
        score = self._call_llm(prompt)
        if score is not None:
            self.scores["methodological"].append(score)

    def evaluate_all(self, translating_eval = False):
        for qna in tqdm(self.dataset):
            # qna = qna["question"], qna["answer"]
            # print(f"Evaluating Q&A their length: {len(self.dataset)}...")

            self._eval_ling_stnd(qna)
            self._eval_cultural(qna)
            self._eval_methodology(qna)
            
            if translating_eval:
              self._eval_trans(qna)

    def get_evaluation_stats(self):
        stats = self.scores
        for category, scores in self.scores.items():
            if scores:
                stats[category] = {
                    "count": len(scores),
                    "mean": round(np.mean(scores), 2),
                    "median": round(np.median(scores), 2),
                    "std_dev": round(np.std(scores), 2),
                    "min": min(scores),
                    "max": max(scores)
                }
            else:
                stats[category] = {
                    "count": 0,
                    "mean": None,
                    "median": None,
                    "std_dev": None,
                    "min": None,
                    "max": None
                }
        return stats
    
    def print_evaluation_stats(self):
      stats = self.get_evaluation_stats()
      
      print("\nEvaluation Statistics:")
      print("="*30)
      
      for category, data in stats.items():
          print(f"\nCategory: {category.capitalize()}")
          print("-"*30)
          print(f"Count: {data['count']}")
          print(f"Mean: {data['mean']}")
          print(f"Median: {data['median']}")
          print(f"Standard Deviation: {data['std_dev']}")
          print(f"Min: {data['min']}")
          print(f"Max: {data['max']}")
          print("-"*30)
      
      print("="*30)

    def get_scores(self):
        return self.scores
