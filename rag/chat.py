from os import getenv
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI


load_dotenv()

qdrant_configs = {
    "url": "http://localhost:6333",
    "collection_name": "node-guide",
}


# user query has to be queried with the same model hence using same embedding model
vector_embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


# prepare same vector DB to process user query
vector_db = QdrantVectorStore.from_existing_collection(
    url=qdrant_configs["url"],
    collection_name=qdrant_configs["collection_name"],
    embedding=vector_embedding_model,
)


user_query = input("Ask me about NodeJS: ")

# perform vector similarity search on user query (get simillar chunks based on what user asked)
search_result = vector_db.similarity_search(query=user_query)

# generate context for LLM based on result received from vector similarity search
CONTEXT = "\n\n\n".join(
    [
        f"Page Content: {result.page_content}\n Page Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
        for result in search_result
    ]
)


# system prompt for LLM
SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who answers user query based on the available context retrieved from a PDF file along page_content and page_number. 
You should only answer the user based on the following context and navigate the user to open the right page number to know more from the provided pdf file. 
You should provide output in markdown format with syntax highlighting if the output contains any code. 
If the user prompt is not not about Nodejs then respond with "I can only help you with NodeJS related queries. Please ask appropriate question"

Context: {CONTEXT}
"""

# make API call to LLM with provided system prompt (which contains context from vector similarity search) and get response back
client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-5-nano",
    reasoning={"effort": "low"},
    input=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)


print(response.output_text)
