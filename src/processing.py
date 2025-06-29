import os
import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
import hydra
from omegaconf import DictConfig
from utils import grade_MATH, grade_NLP
import re

from utils import (
    PromptFormatter, LabelFormatter
)

def extract_balanced_braces(text, start_pos):
    """Helper function to extract content within balanced braces"""
    if start_pos >= len(text) or text[start_pos] != '{':
        return None
    
    brace_count = 0
    i = start_pos
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start_pos + 1:i]  # Return content without braces
        i += 1
    return None

class DataProcessingDebate():
    """Class to process input data into a debate."""
    def __init__(self, config, cot=False):
        """Initialize the DataProcessor.

        Args:
            config (DictConfig): Configuration for the dataset builder.
            tokenizer (AutoTokenizer): Tokenizer for text processing.
        
        """
        self.config = config
        self.cot= cot

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.path, 
        )
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.tokenizer.padding_side = "right"
        
        self.prompt_formatter = PromptFormatter(
            config.dataset.name, 
            cot=self.cot, 
            question_format=config.dataset.format, 
            batch=False
        )

        self.label_formatter = LabelFormatter(
            config.dataset.name,
            batch=False
        )

    def format_prompts(self, item, with_golden_label=True):
        """Format prompts using the tokenizer's chat template.

        Args:
            item (dict): Dictionary containing the instruction, question, and answer.

        Returns:
            str: Formatted prompt.
        """
        if with_golden_label:
            messages = [
                {"role": "system","content": item["instruction"]},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
        else:
            messages = [
                {"role": "system","content": item["instruction"]},
                {"role": "user", "content": item["question"]},
            ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def count_tokens(self, item):
        """Count the number of tokens in the given text.

        Args:
            item (dict): Dictionary containing the text to be tokenized.

        Returns:
            int: Number of tokens.
        """
        return len(
            self.tokenizer(
                item["text"],
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
        )

    def transform(self, dataset, with_golden_label=True):
        """Transform the dataset by formatting prompts and filtering based on token count.

        Args:
            dataset (Dataset): The dataset to be transformed.

        Returns:
            Dataset: Transformed dataset.
        """
        items = []
        for item in dataset:
            instruction, question = self.prompt_formatter.generate(item)
            label, answer = self.label_formatter.generate(item)

            if self.cot:
                items.append(
                        {
                            "instruction": instruction,
                            "question": question,
                            "answer": f"Explanation: {item['supporting_argument']}\nChoice: {label}) {answer}",
                        }
                    )
            else:
                items.append(
                    {
                        "instruction": instruction,
                        "question": question,
                        "answer": f"Choice: {label}) {answer}",
                    }
                )
        df = pd.DataFrame(items)
        df["text"] = df.apply(self.format_prompts, args=(with_golden_label,), axis=1)
        df["token_count"] = df.apply(self.count_tokens, axis=1)

        df = df[df.token_count < self.config.training.max_seq_length]
        df = df.reset_index()
        
        return Dataset.from_pandas(df)

class DataParser():

    def __init__(self, answer_type, dataset_type):
        self.answer_type = answer_type
        self.dataset_type = dataset_type
    
    def grade(self, candidate, answer):
        if self.answer_type == "multiple_choice":
            if self.dataset_type == "QA":
                return grade_NLP(candidate, answer)
            elif self.dataset_type == "Math":
                return grade_MATH(candidate, answer)
            else:
                print("Dataset grading type not recognized")
                return None
        else:
            print("Answer format not recognized")
            return None

    def most_frequent(self, answers):
        answer_set = []
        counts = []

        for answer in answers:
            is_match = False
            for i, candidate_answer in enumerate(answer_set):
                # if the answer is semantically equivalent to something in our current set
                if self.grade(candidate_answer, answer):
                    is_match = True
                    counts[i] = counts[i] + 1
                    break
            # If the answer hasnt been encountered before, add it to the possible answers set
            if not is_match:
                answer_set.append(answer)
                counts.append(1)

        responses = sorted(zip(counts, answer_set))
        return responses[-1][1]
        
    def MC_parse(self, input_str):
        """
        Extracts the content from various answer template formats in a string.
        Supports:
        - \\boxed{answer}
        - Choice: [[answer]]
        - Choice: answer (with newlines)
        - correct answer is answer.
        - Letter) Answer format (A) Answer, B) Answer, etc.)
        
        Args:
            input_str (str): Input string that may contain answer expressions
            
        Returns:
            str or None: Content of the answer, or None if not found
        """
        input_str = input_str.replace('\\text', '')
        input_str = input_str.replace('\\textbf', '')
        input_str = input_str.replace('\\mathbf', '')
        input_str = input_str.replace('bf', '')

        # Method 1: Try to find \\boxed{...} or \\fbox{...} (original logic)
        boxed_result = self._extract_boxed(input_str)
        if not boxed_result is None:
            return boxed_result
        
        # Method 2: Try to find Letter) Answer format
        letter_answer_result = self._extract_letter_answer(input_str)
        if letter_answer_result:
            return letter_answer_result
        
        # Method 4: None of the above answer
        pattern = r'None of the above'
        matches = re.findall(pattern, input_str, re.IGNORECASE)
        
        if matches:
            return 'None of the above'
        return None


    def _extract_boxed(self, input_str):
        """Extract answer from \\boxed{...}"""
        # Find all occurrences of \boxed{ or \fbox{
        boxed_pattern = re.compile(r"[\\]*boxed\s*\{+((?:(?!\\+boxed)[^}])*)(?:\}+|$)", re.VERBOSE | re.DOTALL)
        matches = re.findall(boxed_pattern, input_str)
        if not matches:
            return None
        else:
            return matches[-1]

    def _extract_letter_answer(self, input_str):
        """Extract answer from Letter) Answer format (A) Answer, B) Answer, etc.)"""
        # Pattern to match Letter) followed by answer text
        # Captures until period, newline, or end of string
        pattern = r'([A-Z])\)\s*([^.\n]+?)(?:[.\n]|$)'
        matches = re.findall(pattern, input_str, re.DOTALL)
        
        if matches:
            # Return the last match as "Letter) Answer"
            letter, answer = matches[-1]
            return f"{letter}) {answer.strip()}"
        
        return None