""" Example handler file. """

import os
from typing import Dict
import runpod
import urllib.request

from llama_cpp import Llama, LlamaGrammar, LLAMA_SPLIT_NONE, LLAMA_DEFAULT_SEED

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

repository = os.environ.get("REPOSITORY", "TheBloke/Mistral-7B-v0.1-GGUF")
filename = os.environ.get("FILENAME", "mistral-7b-v0.1.Q5_K_M.gguf")

model_path = "/runpod-volume/" + filename

if os.path.isfile(model_path) == False:
    urllib.request.urlretrieve(
        "https://huggingface.co/" + repository + "/resolve/main/" + filename,
        model_path,
    )

n_gpu_layers = os.environ.get("N_GPU_LAYERS", "-1")
split_mode = os.environ.get("SPLIT_MODE", str(LLAMA_SPLIT_NONE))
main_gpu = os.environ.get("MAIN_GPU", "0")
vocab_only = os.environ.get("VOCAB_ONLY", "False")
use_mmap = os.environ.get("USE_MMAP", "True")
use_mlock = os.environ.get("USE_MLOCK", "False")

seed = os.environ.get("SEED", str(LLAMA_DEFAULT_SEED))
n_ctx = os.environ.get("N_CTX", "512")
n_batch = os.environ.get("N_BATCH", "512")
n_threads = os.environ.get("N_THREADS", "None")
n_threads_batch = os.environ.get("N_THREADS_BATCH", "None")

llm = Llama(
    model_path=model_path,
    n_gpu_layers=int(n_gpu_layers),
    split_mode=int(split_mode),
    main_gpu=int(main_gpu),
    vocab_only=vocab_only.lower() == "true",
    use_mmap=use_mmap.lower() == "true",
    use_mlock=use_mlock.lower() == "true",
    seed=int(seed),
    n_ctx=int(n_ctx),
    n_batch=int(n_batch),
    n_threads=int(n_threads) if n_threads.isdigit() else None,
    n_threads_batch=int(n_threads_batch) if n_threads_batch.isdigit() else None,
)

def handler(job: Dict[str, Dict]):
    job_input = job["input"]

    prompt = job_input.get("prompt", "Q: Name the planets in the solar system? A: ")
    suffix = job_input.get("suffix", None)
    max_tokens = job_input.get("max_tokens", 16)
    temperature = job_input.get("temperature", 0.8)
    top_p = job_input.get("top_p", 0.95)
    min_p = job_input.get("min_p", 0.05)
    typical_p = job_input.get("typical_p", 1.0)
    logprobs = job_input.get("logprobs", None)
    echo = job_input.get("echo", False)
    stop = job_input.get("stop", [])
    frequency_penalty = job_input.get("frequency_penalty", 0.0)
    presence_penalty = job_input.get("presence_penalty", 0.0)
    repeat_penalty = job_input.get("repeat_penalty", 1.1)
    top_k = job_input.get("top_k", 40)
    stream = job_input.get("stream", False)
    seed = job_input.get("seed", None)
    tfs_z = job_input.get("tfs_z", 1.0)
    mirostat_mode = job_input.get("mirostat_mode", 0)
    mirostat_tau = job_input.get("mirostat_tau", 5.0)
    mirostat_eta = job_input.get("mirostat_eta", 0.1)
    model = job_input.get("model", None)
    stopping_criteria = job_input.get("stopping_criteria", None)
    logits_processor = job_input.get("logits_processor", None)
    grammar = job_input.get("grammar", None)
    logit_bias = job_input.get("logit_bias", None)

    return llm(
        prompt=prompt,
        suffix=suffix,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        typical_p=typical_p,
        logprobs=logprobs,
        echo=echo,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repeat_penalty=repeat_penalty,
        top_k=top_k,
        stream=stream,
        seed=seed,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        model=model,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
        grammar=LlamaGrammar.from_string(grammar) if grammar != None else None,
        logit_bias=logit_bias,
    )


runpod.serverless.start({"handler": handler})
