from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#1st Prompt
template1 = PromptTemplate(template="write a detailed report on the topic {topic}", input_variables=["topic"])

#2st Prompt

template2 = PromptTemplate(template="write a 4 point detailed summary on the text {text}", input_variables=["text"])

#String Output Parser
parser = StrOutputParser()

#Chain
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic":"what AI impact on IT nowadays?"})
print(result)