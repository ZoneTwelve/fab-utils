import argparse
import json
import os
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Dataset Generation Script")
parser.add_argument("--batch-size", type=int, default=30, help="Batch size for processing")
parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("--model", type=str, required=True, help="Model path")
parser.add_argument("--quantization", type=str, default=None, help="Quantization type")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Model datatype")
parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager execution")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="GPU memory utilization")
parser.add_argument("--output-file-or-path", type=str, required=True, help="Output file path")
parser.add_argument("--file-format", type=str, choices=["jsonl", "parquet"], default="jsonl", help="Output file format")
parser.add_argument("--file-size", type=int, default=1, help="File size limit in GB before creating a new file")
parser.add_argument("--file-flush", action="store_true", help="Store each batch in real-time")
parser.add_argument("--template", type=str, required=True, help="Prompt template")
parser.add_argument("--system-prompt", type=str, required=True, help="System prompt (file path or string)")
args = parser.parse_args()

# Load system instruction
try:
    if os.path.isfile(args.system_prompt):
        with open(args.system_prompt, "r") as f:
            system_instruction = f.read()
    else:
        system_instruction = args.system_prompt
        print("Warning: Using system prompt as a string instead of a file.")
except Exception as e:
    print(f"Error loading system prompt: {e}")
    exit(1)

# Initialize the vLLM engine
llm = LLM(
    model=args.model,
    tensor_parallel_size=2,
    quantization=args.quantization,
    enforce_eager=args.enforce_eager,
    gpu_memory_utilization=args.gpu_memory_utilization,
    dtype=args.dtype,
)

# Define sampling parameters for generation
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=16384,
)

# Load the dataset
dataset = load_dataset(args.dataset, split="train")

# Prepare for batch processing
batch_questions = []
batch_rows = []
file_index = 0
current_file_size = 0

def write_results(results):
    global file_index, current_file_size
    file_path = f"{args.output_file_or_path}.{file_index}.{args.file_format}"
    
    if args.file_format == "jsonl":
        with open(file_path, "a", encoding="utf-8") as f:
            for result in results:
                json_line = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_line)
                current_file_size += len(json_line.encode('utf-8'))
    elif args.file_format == "parquet":
        df = pd.DataFrame(results)
        df.to_parquet(file_path, index=False, engine='pyarrow', compression='snappy')
        current_file_size = os.path.getsize(file_path)
    
    if current_file_size >= args.file_size * 1024 * 1024 * 1024:
        file_index += 1
        current_file_size = 0

for row in tqdm(dataset, desc="Generating responses"):
    # Prepare chat messages for each row
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": args.template.format(**row)}
    ]

    # Format prompt using the model's chat template
    prompt = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    batch_questions.append(prompt)
    batch_rows.append(row)

    # Process the batch when it reaches the defined size
    if len(batch_questions) >= args.batch_size:
        outputs = llm.generate(batch_questions, sampling_params)
        results = [{**batch_rows[i], "generated_model_output": output.outputs[0].text} for i, output in enumerate(outputs)]
        write_results(results)

        print(batch_rows[0]['question'])
        print(outputs[0].outputs[0].text)

        # Clear the batch
        batch_questions = []
        batch_rows = []

if batch_questions:
    outputs = llm.generate(batch_questions, sampling_params)
    results = [{**batch_rows[i], "generated_model_output": output.outputs[0].text} for i, output in enumerate(outputs)]
    write_results(results)

print(f"Dataset generation complete. Results saved to {args.output_file_or_path}")
