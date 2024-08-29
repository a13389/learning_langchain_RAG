from dotenv import load_dotenv
from langchain_community.chat_models import QianfanChatEndpoint
load_dotenv()

model = QianfanChatEndpoint(model="ERNIE-4.0-8K-Latest", streaming=True)


from langchain.prompts import ChatPromptTemplate

template_multiple = """你是一个脱口秀演员.
Human: 给我讲一个{adjective}主题的故事， 是关于{animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "悲伤", "animal": "小猫"})
print("\n---------------\n")
print(prompt)

result = model.invoke(prompt)
print(result.content)
