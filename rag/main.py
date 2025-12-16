from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# download from https://www.anuragkapur.com/assets/blog/programming/node/PDF-Guide-Node-Andrew-Mead-v3.pdf and save it as  "node-guide.pdf"
pdf_file_path = Path(__file__).parent / "node-guide.pdf"

loader = PyPDFLoader(pdf_file_path)

# parses the pdf file and loads
print(loader.load())
