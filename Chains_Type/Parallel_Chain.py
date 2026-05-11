from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#1st Prompt
prompt1 = PromptTemplate(template="generate short and simple notes for the following text: {topic}",input_variables=["topic"])

#2nd Prompt
prompt2 =PromptTemplate(template="generate 5 short answer from the folllwing  text: {text}",input_variables=["text"])

#3rd prompt
prompt3 = PromptTemplate(template="merge the provided notes and quiz into a single document  Notes: {notes} \n Quiz: {quiz}")

#String Output Parser
parser = StrOutputParser()

Runnabl_Chain = RunnableParallel(
    {
    "notes": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
    }
)

final_chain = prompt3 | model | parser

chain = Runnabl_Chain | final_chain

text = "What is LLM and how it works?"

result = chain.invoke(text)
print(result)
