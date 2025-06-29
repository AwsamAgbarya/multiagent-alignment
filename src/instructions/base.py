# Enforce Multiple choice question format
template_format_mc = """
Your response should consist only of the correct CHOICE, with no explanation included.
Provide your selected CHOICE using the following format:
\\boxed{{Your CHOICE here}}
given at the end of your response.
for example:
\\boxed{{B) The color of the sky is blue.}}
"""


# Intructions promps per dataset
template_instruction_logiqa = """You will be presented with a CONTEXT passage and a corresponding QUESTION with four answer CHOICES. 
Carefully read the passage to understand its content. Then, read the QUESTION and CHOICES thoroughly.
"""

template_instruction_arc_challenge = """You will be presented a QUESTION with multiple answer CHOICES.
Carefully read the QUESTION and CHOICES. 
"""

template_instruction_aqua_rat = """You will be given a QUESTION along with multiple answer CHOICES, involving a math problem that requires step-by-step reasoning to determine the correct answer.
Carefully read the QUESTION and CHOICES. 
"""

template_instruction_openbookqa = """You will be presented a QUESTION with multiple answer CHOICES.
Carefully read the QUESTION and CHOICES. 
"""