from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage

chat = QianfanChatEndpoint(model="ERNIE-4.0-8K-Latest")
messages = [HumanMessage(content="你好，你是谁")]
res = chat.invoke(messages)
print(res.content)

