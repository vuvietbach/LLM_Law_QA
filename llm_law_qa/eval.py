import re
import typer
from llm_law_qa.utils import init_llm, read_jsonl, write_jsonl
import tqdm
class PromptTemplate:
    llm_eval = (
        "Hãy cho biết câu trả lời sau đây có đúng với đáp án. Chỉ cần trả lời 'Đúng' hoăc 'Sai'\n"
        "Đáp án đúng là {}\n"
        "Câu trả lời: {}\n"
    )

app = typer.Typer()
class Pattern:
    ground_truth_answer = r"Chọn đáp án ([ABCD])"
    direct_answer = [r"Đáp án đúng: ([ABCD])", r"Do đó,? chọn đáp án: ([ABCD])"]

def process_answer(answer_path, pattern_type):
    pattern = Pattern.direct_answer[pattern_type]
    answers = read_jsonl(answer_path)
    answer1 = {}
    for answer in answers:
        try:
            answer1[answer["question_id"]] = re.search(pattern, answer["answer"]).group(1)
        except:
            answer1[answer["question_id"]] = "0"
    return answer1

def process_ground_truth(ground_truth_path):
    ground_truth = read_jsonl(ground_truth_path)
    ground_truth1 = {}
    for gt in ground_truth:
        ground_truth1[gt["id"]] = re.search(Pattern.ground_truth_answer, gt["answer"]).group(1)
    return ground_truth1

@app.command()
def eval_direct(ground_truth_path, answer_path, pattern_type: int = 0):
    ground_truth = process_ground_truth(ground_truth_path)
    answer = process_answer(answer_path, pattern_type)
    num_correct = 0
    for id in ground_truth:
        if id in answer and answer[id] == ground_truth[id]:
            num_correct += 1
        
    acc = num_correct / len(ground_truth)
    log_path = 'log/eval_answer.txt'
    with open(log_path, 'a') as f:
        f.write("Answer path: {}\n".format(answer_path))
        f.write("Ground truth path: {}\n".format(ground_truth_path))
        f.write("Accuracy: {}\n".format(acc))
        f.write("\n")

@app.command()
def eval(ground_truth_path, answer_path):
    ground_truth = read_jsonl(ground_truth_path)
    ground_truth = {gt['id']: gt['correct_answer'] for gt in ground_truth}
    
    answers = read_jsonl(answer_path)
    answers = {answer['question_id']: answer['answer'] for answer in answers}

    num_correct = 0
    for id in tqdm.tqdm(ground_truth):
        if id in answers and answers[id] == ground_truth[id]:
            num_correct += 1         
    acc = num_correct / len(ground_truth)
    
    log_path = 'log/eval_answer.txt'
    with open(log_path, 'a') as f:
        f.write("Answer path: {}\n".format(answer_path))
        f.write("Ground truth path: {}\n".format(ground_truth_path))
        f.write("Accuracy: {}\n".format(acc))
        f.write("\n")

@app.command()
def deep_eval(ground_truth_path, answer_path):
    ground_truth = process_ground_truth(ground_truth_path)
    answers = read_jsonl(answer_path)
    answers = {answer['question_id']: answer['answer'] for answer in answers}
    llm = init_llm()

    num_correct = 0
    for id in tqdm.tqdm(ground_truth):
        if id in answers:
            answer = answers[id]
            prompt = PromptTemplate.llm_eval.format(ground_truth[id], answer)
            result = llm.invoke(prompt)
            write_jsonl('tmp/eval_result.jsonl', [{"question_id": id, "result": result}])
            
            if "Đúng" in result:
                num_correct += 1        
        
    acc = num_correct / len(ground_truth)
    log_path = 'tmp/eval_answer.txt'
    with open(log_path, 'a') as f:
        f.write("Answer path: {}\n".format(answer_path))
        f.write("Ground truth path: {}\n".format(ground_truth_path))
        f.write("Accuracy: {}\n".format(acc))
        f.write("\n")

if __name__ == '__main__':
    app()