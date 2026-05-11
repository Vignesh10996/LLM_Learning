from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#1st Prompt
Prompt1 = PromptTemplate(template="generate 3 detailed report on a topic {topic}", input_variables=["topic"])

#2st Prompt
prompt2 = PromptTemplate(template="generate a 3 point summary on following text{Prompt1}",input_variables=["text"])

#String Output Parser
parser =StrOutputParser()

#Chain
chain = Prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic":"What is LLM and how it works?"})   
print(result)