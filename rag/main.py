from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# download from https://www.anuragkapur.com/assets/blog/programming/node/PDF-Guide-Node-Andrew-Mead-v3.pdf and save it as  "node-guide.pdf"
pdf_file_path = Path(__file__).parent / "node-guide.pdf"

# parses the pdf file and loads texts in the program
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()


# split the document into smaller text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
text_chunks = text_splitter.split_documents(documents)

print(text_chunks)
