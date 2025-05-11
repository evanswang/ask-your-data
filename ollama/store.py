# import sys
# sys.path.append('../..')
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


current_path = Path.cwd()
root_path = current_path.parent
data_path = root_path / "data"

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader(data_path / "survey.pdf"),
    PyPDFLoader(data_path / "my_pdf.pdf"),
    PyPDFLoader(data_path / "ebeebc7b-170e-4c21-9ba7-3e767ac5fcb2-vaf.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

print(len(splits))

# from langchain_community.embeddings import OpenAIEmbeddings
# embedding = OpenAIEmbeddings()

from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_community.vectorstores import Chroma

persist_directory = str(root_path / 'docs' / 'chroma')

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

question = "what is the house address"
docs = vectordb.similarity_search(question,k=3)
print(docs[0].page_content)
vectordb.persist()
