from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal
from pydantic import BaseModel,Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#schema
class Review(BaseModel):
    key_theams: Annotated[list[str], Field(description="must write 3 key themes")]
    summary: Annotated[str ,Field(description="must write a summary of the review in 20 words")]
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="must write the sentiment of the review,either positive, negative or neutral")

strutured_model = model.with_structured_output(Review, strict=True)

propmts ="""
Google Pixel phones are a series of smartphones known for their clean Android experience and strong integration
with Google services. They are designed to deliver smooth performance with a simple and user-friendly interface.
One of the biggest highlights of Pixel phones is their excellent camera quality, which uses advanced software to produce 
detailed and vibrant photos. These devices often receive Android updates earlier than other smartphones, ensuring users 
have the latest features and security patches. Pixel phones also include smart AI features that help with tasks like voice 
commands, photo editing, and real-time assistance. The design of Pixel phones is usually minimal and elegant, appealing to
users who prefer a simple look. They are suitable for everyday use such as browsing, calling, and social media. Although they
may not always have the most powerful hardware, they perform efficiently due to optimized software. Battery life is generally 
decent but may vary depending on usage. Overall, Pixel phones are a great choice for people who value camera quality, software 
experience, and timely updates.
"""
result = strutured_model.invoke(propmts)
print(result)
