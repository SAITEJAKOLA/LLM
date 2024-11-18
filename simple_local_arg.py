import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from dotenv import load_dotenv
from tqdm.auto import tqdm
import re
import subprocess
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone
import random
import pandas as pd
from transformers import BitsAndBytesConfig


# Load environment variables
load_dotenv()

# Check for MPS (Metal Performance Shaders) on macOS
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"[INFO] Using device: {device}")

# GPU memory check (adjusted for macOS)
def get_gpu_memory_mac():
    try:
        output = subprocess.check_output(["sysctl", "hw.memsize"])
        total_memory_bytes = int(output.decode().strip().split(": ")[1])
        reserved_memory_bytes = 2 * 1024 * 1024 * 1024  # Assume 2GB reserved for OS
        available_memory_bytes = total_memory_bytes - reserved_memory_bytes
        available_memory_gb = available_memory_bytes / (1024 ** 3)
        return round(available_memory_gb, 2)
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return None

gpu_memory_gb = get_gpu_memory_mac()
print(f"[INFO] Available GPU memory: {gpu_memory_gb} GB")

# Model selection based on GPU memory
if gpu_memory_gb < 5.1:
    print("[WARNING] Low GPU memory detected. You may need to use CPU or smaller models.")
    model_id = "google/gemma-2b-it"
    use_quantization_config = True
elif gpu_memory_gb < 8.1:
    print("[INFO] Recommended model: Gemma 2B in 4-bit precision.")
    model_id = "google/gemma-2b-it"
    use_quantization_config = True
elif gpu_memory_gb < 19.0:
    print("[INFO] Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
    model_id = "google/gemma-2b-it"
    use_quantization_config = False
else:
    print("[INFO] Recommended model: Gemma 7B in float16 precision.")
    model_id = "google/gemma-7b-it"
    use_quantization_config = False

print(f"[INFO] Using model: {model_id}")

# Tokenizer and model setup
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

if use_quantization_config:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True
    )
else:
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

# Send model to device
llm_model.to(device)

# Check model size
def get_model_size(model):
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem_total = mem_params + mem_buffers
    return round(mem_total / (1024 ** 3), 2)

model_size_gb = get_model_size(llm_model)
print(f"[INFO] Model size: {model_size_gb} GB")

# Set up Hugging Face embeddings
    # Load embeddings and vectorstore
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ.get("HUGGING_API_KEY"),
    model_name="sentence-transformers/all-mpnet-base-v2",
)
Pinecone(api_key = os.environ['PINECONE_API_KEY'])
index_name = "llmchat"
vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Tokenize and generate responses
def tokenize_input(prompt, context, chat_history):
    combined_input = f"Context: {context}\n\n Chat_History: {chat_history}\n\n Question: {prompt}"
    return tokenizer(combined_input, return_tensors="pt").to(device)

def generate_response(input_ids):
    outputs = llm_model.generate(input_ids=input_ids["input_ids"], max_new_tokens=512)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Full Response: ", full_response)
    answer = full_response.split("Answer:")[-1].strip()
    print("Answer: ", answer )
    return answer

# Retrieve and process queries
def retrieve_answers_with_llm_model(query, chat_history):
    results = vectorstore.similarity_search(query, k=4)
    context = "\n".join([doc.page_content for doc in results])
    input_ids = tokenize_input(query, context, chat_history)
    response = generate_response(input_ids)
    return response

# # Test a query
# input_query = "What are the priorities of 2023-2024 budget?"
# response = retrieve_answers_with_llm_model(input_query)
# print(f"Response:\n{response}")
