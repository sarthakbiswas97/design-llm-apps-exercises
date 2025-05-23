import os
import ssl
import nltk
import certifi

#set SSL_CERT_FILE via your shell, do it here:
os.environ["SSL_CERT_FILE"] = certifi.where()


# Ensure NLTK models are present, download if missing:
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# Now your PDF‐reading code...
from unstructured.partition.auto import partition

def read_pdf_with_unstructured(file_path):
    try:
        elements = partition(filename=file_path)
        return "\n\n".join(str(el) for el in elements)
    except Exception as e:
        return f"Error reading PDF: {e}"

if __name__ == "__main__":
    print(read_pdf_with_unstructured("../docs/sample.pdf"))
