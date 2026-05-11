from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

chat_template = ChatPromptTemplate([
    ('system',"you are a helpful {domain} expert."),
    ('human',"EXplain in simple terms, the concept of {topic}")
])

propmt = chat_template.invoke({
    'domain': "quntum physis",
    'topic':"wormhole"
})

print(propmt)
