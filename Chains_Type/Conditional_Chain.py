from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

class Report(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="the sentiment of the feedback provided")

parser2 = PydanticOutputParser(pydantic_object=Report)

#1st Prompt
prompt1 = PromptTemplate(template="Classify the sentiment of the following feedback as positive, negative, or neutral:{feedback}and provide the response in following format:{response_format} ",
                          input_variables=["feedback"],
                          partial_variables={"response_format": parser2.get_format_instructions()}
)

classfier_chain = prompt1 | model | parser2

#2nd Prompt
prompt2 = PromptTemplate(template="write an appropriate response to this positive feedback:{feedback} ",
                          input_variables=["feedback"]
                          )
#3rd Prompt
prompt3 = PromptTemplate(template="write an appropriate response to this negative feedback:{feedback} ",
                          input_variables=["feedback"]
                          )
#4th Prompt
prompt4 = PromptTemplate(template="write an appropriate response to this neutral feedback:{feedback} ", 
                          input_variables=["feedback"]
                          )
branch_Chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    (lambda x: x.sentiment == 'neutral', prompt4 | model | parser),
    RunnableLambda(lambda x: "Invalid sentiment classification")
    
)

#Final Chain
final_chain = classfier_chain | branch_Chain

result = final_chain.invoke({"feedback": "I am notlove the product, it's not amazing!"})
print(result)
