""" Example handler file. """

import os
import runpod
import urllib.request

from llama_cpp import Llama

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

llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
)

def handler(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "Q: Name the planets in the solar system? A: ")

    return llm(prompt)


runpod.serverless.start({"handler": handler})
