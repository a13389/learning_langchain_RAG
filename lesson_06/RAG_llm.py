# 导入 embedding 的大模型
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import QianfanChatEndpoint



load_dotenv()
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma

LLM = QianfanChatEndpoint(model="ERNIE-4.0-8K-Latest",streaming=True)

# 加载文本并切分
qianfan_embeddings = QianfanEmbeddingsEndpoint(model="bge_large_zh", endpoint="bge_large_zh")
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
file_path = os.path.join(parent_dir, "document_file", "生活是美好的.txt")

"""
TextLoader: 用于加载文本文件，如.txt、.md等。
RecursiveCharacterTextSplitter: 用于将文本拆分为块，以供后续处理。
"""
loader = TextLoader(file_path)
docs = loader.load()
# print(docs)
# exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每块的最大字符数
    chunk_overlap=2,  # 块之间的重叠字符数
)


documents_splitted = text_splitter.split_documents(docs)

# print(len(
#     documents_splitted
# ))
# print(documents_splitted[3])
# exit()

persist_directory = "db"

vectordb = Chroma.from_documents(documents=documents_splitted, embedding=qianfan_embeddings, persist_directory=persist_directory)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

user_query = "为什么牙痛反而要高兴？"
docs = retriever.get_relevant_documents(user_query)

context = "\n".join([doc.page_content for doc in docs])


prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="基于以下内容：\n{context}\n回答问题：{query}"
)

prompt_template = prompt_template | LLM


result = prompt_template.invoke({"context": context, "query": user_query})

print(result.content)
