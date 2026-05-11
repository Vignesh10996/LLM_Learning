from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder

#chat_template
chat_template=ChatPromptTemplate([
    'system', "you are a very helpful customer agent",
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')

])

#load chat_history
chat_history =[]
with open("chatbot_History.txt") as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query":"where is my refund?"
})

print(prompt)