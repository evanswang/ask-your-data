from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


print(f"******************** starting app.py ********************")
# Initialize embedding and vector DB
current_path = Path(__file__).resolve()
print(f"******************** current_path: {current_path} ********************")

persist_directory = current_path.parent.parent / 'docs' / 'chroma'
print(f"******************** persist_directory: {persist_directory} ********************")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=str(persist_directory), embedding_function=embedding)
vector_count = vectordb._collection.count()
print(f"******************** starting vectordb {vector_count} ********************")

# Initialize LLM and memory
llm = ChatOllama(model="llama3")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
print(f"******************** starting ConversationBufferMemory ********************")

# Set up ConversationalRetrievalChain
retriever = vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
# print(f"******************** starting qa ********************")
#
# question = "who is my partner?"
# result = qa({"question": question})
# print(f"******************** {result["answer"]} ********************")