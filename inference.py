from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import multiprocessing
import json, os, re
from datasets import Dataset

os.environ["HF_TOKEN"] = "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"
instruct_prompt = r"Answer the question based on the following example:"
example1 = r"""Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Answer: First find how many liters of the seawater are salt: 2 liters * 20% = 0.4 liters Then multiply that amount by 1000 ml/liter to find the number of ml of salt Jack gets: 0.4 liters * 1000 ml/liter = 400 ml."""
example2 = r"""Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name."""
few_shot_cot_prompt = instruct_prompt + '\n' + example2 + f'\nQuestion: '  #'\n' + example1

def tokenize(sample):
    answer_text = sample['answer'].split('####')[-1].strip()
    sample["few_shot_cot_question"] = few_shot_cot_prompt + sample['question']
    sample["answer_text"] = f"The answer is {answer_text}."
    return sample

train_path = "openai/gsm8k"
dataset_ = load_dataset(train_path, "main")["train"]
dataset_ = dataset_.map(tokenize, num_proc=16)
questions = dataset_["few_shot_cot_question"]
answers = dataset_["answer_text"]

NUM_GPUS = 4
split_list = lambda l, n: [l[i * len(l) // n: (i + 1) * len(l) // n] for i in range(n)]
def run_inference_one_gpu(gpu_id, question_list, answer_list):
    #os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"][gpu_id] #str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[gpu_id]
    return generate_rational(question_list, answer_list)

def run_inference_multi_gpu(questions, answers):
    split_questions = split_list(questions, NUM_GPUS)
    split_answers = split_list(answers, NUM_GPUS)
    inputs = [(i, p, split_answers[i]) for i, p in enumerate(split_questions)]
    with multiprocessing.Pool(processes=NUM_GPUS) as pool:
        results = pool.starmap(run_inference_one_gpu, inputs)
    outputs = []
    for result in results:
        outputs.extend(result)
    return outputs
model_name = "Q_models/tt_3/debug2"
#model_name = "/home/swb9572/generate_z_zero/Q_models/debug/debug2"  # "Qwen/Qwen2.5-Math-1.5B"#"facebook/opt-125m"#"google/gemma-2-2b-it"#"deepseek-ai/deepseek-math-7b-rl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    seed=42,
    max_tokens=512,
    min_tokens=50,
    n=1,
    # frequency_penalty=1.0,
    stop_token_ids=[tokenizer.eos_token_id],
    stop=['Question:'],
)

def generate_rational(few_shot_questions, answers):
    llm = LLM(model=model_name, tokenizer=model_name, dtype="bfloat16", seed=42, gpu_memory_utilization=0.9)
    rational = llm.generate(few_shot_questions, sampling_params, use_tqdm=True)
    # print("rational:", rational)
    rational_answer = [rational[i].outputs[0].text + answer_text for i, answer_text in enumerate(answers)]
    return rational_answer

if __name__ == "__main__":
    rational_answer = run_inference_multi_gpu(questions, answers)
    num_train_data = len(dataset_)
    gathered_data = []
    for i in range(num_train_data):
        tmp_data = {"question": dataset_[i]["question"], "answer": dataset_[i]["answer"],
                    "rational_answer": rational_answer[i]}
        gathered_data.append(tmp_data)

    with open("./out.json", "w", encoding="utf8") as f:
        json.dump(gathered_data, f, ensure_ascii=False)
    dataset = Dataset.from_list(gathered_data)
    dataset.push_to_hub("ZhangShenao/gemma_gsm8k_rational_answer", private=False)