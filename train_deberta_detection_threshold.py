import subprocess
packages = ["torch", "transformers", "rouge_score", "bert_score", "datasets", "accelerate", "sentencepiece"]
for pkg in packages:
    try:
        mod = __import__(pkg.replace("-", "_"))
        version = getattr(mod, "__version__", "installed")
        print(f" {pkg}: {version}")
    except ImportError:
        print(f" {pkg}: NOT FOUND")
