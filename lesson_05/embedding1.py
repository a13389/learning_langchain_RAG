from dotenv import load_dotenv

load_dotenv()
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

embeddings = QianfanEmbeddingsEndpoint(
    # model="bge_large_zh",
    # endpoint="bge_large_zh"
)

embedded = embeddings.embed_documents(
    [
        "大家好!",
        "你干嘛!",
        "学习两年半",
        "唱跳RAP",
        "我喜欢篮球!"
        ]
)
print(len(embedded))
print("一共是",len(embedded[0]),"维度")

for r in embedded:
    print(r[:8])