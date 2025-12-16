from pathlib import Path
from os import getenv
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


load_dotenv()

# download from https://www.anuragkapur.com/assets/blog/programming/node/PDF-Guide-Node-Andrew-Mead-v3.pdf and save it as  "node-guide.pdf"
pdf_file_path = Path(__file__).parent / "node-guide.pdf"
qdrant_configs = {
    "url": "http://localhost:6333",
    "collection_name": "node-guide",
}


print("Loading PDF Document")

# parses the pdf file and loads texts in the program
loader = PyPDFLoader(pdf_file_path)
documents = loader.load()


print("Splitting text content from PDF")
# split the document into smaller text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
text_chunks = text_splitter.split_documents(documents)

# creating vector embeddings
vector_embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


print("Generating vector embeddings from text content")
vector_store = QdrantVectorStore.from_documents(
    documents=text_chunks,
    embedding=vector_embedding_model,
    url=qdrant_configs["url"],
    collection_name=qdrant_configs["collection_name"],
    api_key=getenv("OPENAI_API_KEY"),
)

print("Vector embeddings generation completed")
