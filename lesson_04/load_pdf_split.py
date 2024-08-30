# -*- coding: utf-8 -*

import os

from langchain_text_splitters import (RecursiveCharacterTextSplitter,CharacterTextSplitter)

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

from langchain_community.document_loaders import PyPDFLoader


file_path = os.path.join(parent_dir, "document_file", "费用管理制度.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()


# 使用 RecursiveCharacterTextSplitter 进行切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每个块的最大字符数
    chunk_overlap=20,  # 相邻块之间的字符重叠数
    separators=["\n\n", ".", " "]  # 按段落、句子和空格递归切分
)

# 将 PDF 文档进行切分
split_docs = text_splitter.split_documents(documents)

# 输出切分后的块
print("切分后的块数：", len(split_docs))
for i, chunk in enumerate(split_docs):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")