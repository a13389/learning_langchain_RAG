import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
file_path = os.path.join(parent_dir, "document_file", "生活是美好的.txt")


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
"""
TextLoader: 用于加载文本文件，如.txt、.md等。
RecursiveCharacterTextSplitter: 用于将文本拆分为块，以供后续处理。
"""
loader = TextLoader(file_path)
docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,  # 每块的最大字符数
#     chunk_overlap=20,  # 块之间的重叠字符数
#     separators=["\n\n", ".", " "]  # 按段落、句子和空格递归切分
# )

text_splitter2 = CharacterTextSplitter(
    chunk_size=80,  # 每块的最大字符数
    chunk_overlap=5,  # 块之间的重叠字符数
    separator="\n"  # 用于拆分文本的分隔符
)

docs = text_splitter2.split_documents(docs)
print(len(docs))

for i, doc in enumerate(docs):
    print(f"文档块 {i+1}:\n{doc.page_content}\n{'-'*40}")
