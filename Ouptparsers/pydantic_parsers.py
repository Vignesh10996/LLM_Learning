from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field    

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.9)

class Report(BaseModel):
    name: str = Field(description="the person's full name")
    age: int = Field(gt=18, lt=100, description="the person's age must be greater than 18 and less than 100")
    city: str = Field(description="the city where the person lives")

parser = PydanticOutputParser(pydantic_object=Report)

template = PromptTemplate(
    template="Give me the name, age and city of the functional {place} person. Make sure the age is greater than 18 and less than 100. Return the output in the following format:\n{response_format}",
    input_variables=["place"],
    partial_variables={"response_format": parser.get_format_instructions()},
)

chain = template | model | parser
result = chain.invoke({"place": "kovilpatti"})
print(result)
