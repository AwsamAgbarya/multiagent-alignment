import instructions.base as instructions_base
import instructions.base_cot as instructions_base_cot
import re
import logging
import gc
import os
import datasets
from collections import defaultdict
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from typing import Optional

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"

class PromptFormatter:
    """A class to format prompts for different datasets."""

    def __init__(self, dataset_name, cot=True, question_format='multiple_choice', batch=False):
        """
        Initialize the PromptFormatter object.
        
        Parameters:
            dataset_name (str): The name of the dataset.
        """
        self.dataset_name = dataset_name
        self.question_format = question_format
        self.cot = cot
        self.batch = batch

        if self.cot:
            self.instructions = instructions_base_cot
        else:
            self.instructions = instructions_base

        self.prompt_dataset_dict = {
            'aqua_rat': self._prompt_aqua_rat,
            'arc_challenge': self._prompt_arc_challenge,
            'logiqa': self._prompt_logiqa,
            'openbookqa': self._prompt_openbookqa,
        }
    
    def generate(self, sample):
        """
        Generate a formatted prompt based on the dataset.
        
        Parameters:
            sample (dict): The sample data to generate the prompt.

        Returns:
            list: A list of messages representing the formatted prompt.
        """
        if self.dataset_name in self.prompt_dataset_dict:
            transformation = self.prompt_dataset_dict[self.dataset_name]
            if self.batch:
                return batch_processing(sample, transformation)
            else: 
                return transformation(sample)
        else:
            raise ValueError("Invalid dataset")
    
    def _format_system_content(self, template):
        if self.question_format == 'multiple_choice':
            system_content = template + self.instructions.template_format_mc
        elif self.question_format == 'open_question':
            system_content = template + self.instructions.template_format_oq
        else:
            system_content = template
        return system_content
    
    def _prompt_aqua_rat(self, sample):
        question= sample["question"]
        rationale= sample["rationale"]
        options = "".join(f"{value} " for i, value in enumerate(sample["options"]))

        instruction = self._format_system_content(self.instructions.template_instruction_aqua_rat)
        question  = "Question:\n" + question + "\n\nChoices:\n"+ options +" \n"
        
        return instruction, question
    
    def _prompt_arc_challenge(self, sample):
        question= sample["question"]
        options = "".join(f"{chr(ord('A') + i)}) {value} " for i, value in enumerate(sample["choices"]["text"]))
        
        instruction = self._format_system_content(self.instructions.template_instruction_arc_challenge)
        question  = "Question:\n" + question + "\n\nChoices:\n"+ options +" \n"

        return instruction, question

    def _prompt_logiqa(self, sample):
        context = sample["context"]
        question= sample["question"]
        options = "".join(f"{chr(ord('A') + i)}) {value}\n" for i, value in enumerate(sample["options"]))

        instruction = self._format_system_content(self.instructions.template_instruction_logiqa)
        question = "Context:\n" + context + "\n\nQuestion:\n" + question + "\n\nChoices:\n"+ options +" \n"
    
        return instruction, question
    
    def _prompt_openbookqa(self, sample):
        question= sample["question_stem"]
        options = "".join(f"{chr(ord('A') + i)}) {value} " for i, value in enumerate(sample["choices"]["text"]))
        
        instruction = self._format_system_content(self.instructions.template_instruction_openbookqa)
        question  = "Question:\n" + question + "\n\nChoices:\n"+ options +" \n"

        return instruction, question

        
class LabelFormatter:
    """A class to format labels for different datasets."""

    def __init__(self, dataset_name, batch=False):
        """
        Initialize the LabelFormatter object.
        
        Parameters:
            dataset_name (str): The name of the dataset.
        """
        self.dataset_name = dataset_name
        self.batch = batch
        self.label_dataset_dict = {
            'aqua_rat': self._label_aqua_rat,
            'arc_challenge': self._label_arc_challenge,
            'logiqa': self._label_logiqa,
            'openbookqa': self._label_openbookqa,
        }
    
    def generate(self, sample):
        """
        Generate a formatted label based on the dataset.
        
        Parameters:
            sample (dict): The sample data to generate the label.

        Returns:
            int or None: The label or None if not applicable.
        """
        if self.dataset_name in self.label_dataset_dict:
            transformation = self.label_dataset_dict[self.dataset_name]
            if self.batch:
                return batch_processing(sample, transformation, pivot=True)
            else:
                return transformation(sample)

        else:
            raise ValueError("Invalid dataset")
    
    def _label_aqua_rat(self, sample):
        label = sample['correct']
        answer = sample['options'][ord(label) - ord('A')]
        index = answer.find(')')
        if index != -1:
            answer = answer[index + 1:]
        return label, answer

    def _label_arc_challenge(self, sample):
        label = sample['answerKey']
        labels = sample['choices']['label']
        answer = sample["choices"]["text"][labels.index(label)]
        return label, answer
    
    def _label_logiqa(self, sample):
        label = sample['label'].upper()
        answer = sample["options"][ord(label) - ord('A')]
        return label, answer

    def _label_openbookqa(self, sample):
        label = sample['answerKey']
        answer = sample["choices"]["text"][ord(label) - ord('A')]
        return label, answer
    

class OutputFormatter:
    """A class to format output based on specified format type."""

    def __init__(self, format_type="cot"):
        """
        Initialize the OutputFormatter object.
        
        Parameters:
            format_type (str): The type of output format.
        """
        self.format_type = format_type
        self.output_dict = {
            'cot': self._parser_llm_explain,
            'judge': self._parser_judgment
        }

    def generate(self, sample):
        """
        Generate formatted output based on the specified format type.
        
        Parameters:
            data_str (str): The string data to format.

        Returns:
            dict: A dictionary containing the formatted output.
        """
        if self.format_type in self.output_dict:
            dict_list = [self.output_dict[self.format_type](item) for item in sample]
            return transform_dict_list(dict_list)
        else:
            raise ValueError("Invalid format type")


    def _parser_llm_explain(self, text):        
        try:
            val_explain = extract_explanation(text)
            val_answer = extract_choice(text)
        
            if val_answer and len(val_answer)>1:
                val_answer = try_fixing_answer_formatting(val_answer)

            if val_explain and val_answer:
                return {"explanation": val_explain, "answer": val_answer} 
            else:
                return {"explanation": None, "answer": None} 
    
        except Exception as e:
            return {"explanation": None, "answer": None} 

    def _parser_judgment(self, text):        
        try:
            critic = extract_judgment_explanation(text)
            rating = extract_judgment_rating(text)
    
            if (critic is not None) and (rating is not None):
                return {"critic": critic, "rating": rating} 
            else:
                return {"critic": None, "rating": None} 
    
        except Exception as e:
            return {"critic": None, "rating": None} 
    

def try_fixing_answer_formatting(answer_string):
    """
    Extracts the option letter (e.g., 'A') from an answer string like:
    - '[[A) Some text]]'
    - 'Choice C)'
    - 'Answer: D)'
    - 'B) Text'
    """
    match = re.search(r'\b([A-Z])\)', answer_string)
    
    if match:
        return match.group(1)
    else:
        return None

def extract_explanation(text, dtype='str'):
    """Extract Explanation from response."""
    try:    
        value = text.split(f"Choice:")[0].strip()
    except Exception as e:
        return None

    if value:
        if dtype=='str':
            return value
    else:
        return None

def extract_choice(text, dtype='str'):
    """Extract Choice from response."""
    try:    
        value = text.split(f"Choice:")[-1].strip()
    except Exception as e:
        return None

    if value:
        if dtype=='str':
            return value
    else:
        return None

def extract_judgment_explanation(text, dtype='str'):
    """Extract Judgment justification from response."""
    try:    
        value = text.split(f"Rating:")[0].strip()
    except Exception as e:
        return None

    if value:
        if dtype=='str':
            return value
    else:
        return None

def extract_judgment_rating(text):
    """Extracts a judgment rating (0–10) from a given text, handling various formats."""
    match = re.search(r'(?i)rating[:\s]*\[*\[*\s*(\d{1,2})\s*\]*\]*', text)
    if match:
        rating = int(match.group(1))
        if 0 <= rating <= 10:
            return float(rating)
    return None

def extract_selected_choice(text, model_answer):
    """extract the full text of an option given the text and the option letter"""
    text = text.split('Choices:')[1]
    # Pattern matches even if repeated letters (like C)C))
    # Find all choices like A) ..., B) ..., etc.
    choices = re.findall(r'([A-Z])\)\s*(.*?)\s*(?=[A-Z]\)|$)', text, re.DOTALL)

    # Create a dict to map letter -> choice text
    choice_dict = {letter: text.strip() for letter, text in choices}

    # Get the extracted answer for the model's letter
    return f"{model_answer}) {choice_dict.get(model_answer, '')}"

def trim_content(text):
    """Trim text"""
    lines = text.split('\n')
    trimmed_lines = [line.lstrip() for line in lines]
    trimmed_text = '\n'.join(trimmed_lines)
    return trimmed_text.strip()

def add_index(item, idx):
        item['index'] = idx
        return item

def batch_processing(samples, transformation, pivot= None):
    keys = [k for k in samples.keys()]
    batch_size = len(samples[keys[0]])
    output = []
    for i in range(batch_size):
        list_of_tuples = [(k, samples[k][i]) for k in keys]
        item = dict(list_of_tuples)
        item = transformation(item)
        output.append(item)
    if pivot:
        keys, values = zip(*output)
        return [list(keys), list(values)]
    else:
        return  output 

def transform_dict_list(dict_list):
    if not dict_list:
        return {}

    transformed_dict = {key: [] for key in dict_list[0]}
    for item in dict_list:
        for key, value in item.items():
            transformed_dict[key].append(value)
    
    return transformed_dict

def remove_duplicates(dataset, column_name):
    """Remove the second occurrence when a column value is the same."""
    seen = defaultdict(bool)
    unique_indices = []
    
    for idx, row in enumerate(dataset):
        value = row[column_name]
        if not seen[value]:
            unique_indices.append(idx)
            seen[value] = True
            
    return dataset.select(unique_indices)
    
def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
    
def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_MATH(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

def grade_NLP(candidate, answer):
    
    c = candidate.split(")")
    a = answer.split(")")
    if c[0] == a[0] or ' '.join(c[1:]) == ' '.join(a[1:]):
        return True
    return False