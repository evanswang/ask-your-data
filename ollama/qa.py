from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

current_path = Path.cwd()
persist_directory = current_path.parent / 'docs' / 'chroma'
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=str(persist_directory), embedding_function=embedding)

print(vectordb._collection.count())

question = ("Any information about my partner?")
docs = vectordb.similarity_search(question,k=3)
print(docs[0].page_content)

from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="llama3")

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
)
result = qa_chain({"query": question})
print(result["result"])