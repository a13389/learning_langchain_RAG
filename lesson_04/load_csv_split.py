import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
file_path = os.path.join(parent_dir, "document_file", "wukong.csv")

loader = CSVLoader(file_path=file_path)
data = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,  # 每块的最大字符数
    chunk_overlap=5,  # 块之间的重叠字符数
    separator="\n"  # 用于拆分文本的分隔符
)


docs = text_splitter.split_documents(data)
print(len(docs))

for i, doc in enumerate(docs):
    print(f"文档块 {i+1}:\n{doc.page_content}\n{'-'*40}")