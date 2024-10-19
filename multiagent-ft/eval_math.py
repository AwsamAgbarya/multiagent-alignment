import json
import numpy as np
import sympy.parsing.latex as latex
from grader import grade_answer

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def most_frequent(answers):
    counter = 0

    answer_set = []
    counts = []

    for answer in answers:
        is_match = False
        for i, candidate_answer in enumerate(answer_set):
            if grade_answer(candidate_answer, answer):
                is_match = True
                counts[i] = counts[i] + 1
                break

        if not is_match:
            answer_set.append(answer)
            counts.append(1)

    responses = sorted(zip(counts, answer_set))
    print(responses)

    return responses[-1][1]


def parse_answer(input_str):

	return remove_boxed(last_boxed_only_string(input_str))

if __name__ == "__main__":
    file_path = None
    response = json.load(open(file_path, "r"))
    correct = []

    consensus = []

    for k, v in response.items():
        response_list, solution = v
        solution_val = parse_answer(solution)
        responses = []

        for response in response_list:
            response_val = parse_answer(response[3]['content'])

            if response_val is not None:
                responses.append(response_val)
        try:
            response = most_frequent(responses)
        except:
            continue

        for r in responses:
            if r == response:
                consensus.append(1)
            else:
                consensus.append(0)

        print("mean consensus: ", np.mean(consensus))

        if grade_answer(response, solution_val):
            correct.append(1)
        else:
            correct.append(0)

        print("correct accuracy: ", np.mean(correct), np.std(correct) / len(correct) ** 0.5)
