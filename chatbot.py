
#! Imports
from langchain_chroma import Chroma

from langchain_community.chat_message_histories import FileChatMessageHistory

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain.chat_models import init_chat_model

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, MessagesState

from langgraph.checkpoint.memory import MemorySaver

from typing import List, Sequence
from typing_extensions import Annotated, TypedDict

from pydantic import BaseModel, Field


import dotenv
import os

#! Output structure
class Classification(BaseModel):
    summaryField: str = Field(
        description="A concise one-sentence summary of the conversation, capturing the most crucial information or the main point discussed. "
        "The summary should be in the format: 'text'. "
        "Include only key decisions or important names mentioned. "
        "Avoid repetitions and ensure clarity. "
        "Optimize for brevity and representativeness of the conversation."
    )
    # amélioration possible 
    # language: str = Field(description="The language the text is written in", enum=["en", "fr"])

#! Configuration
dotenv.load_dotenv()
llm = ChatMistralAI()
structured_llm = init_chat_model("mistral-large-latest", model_provider="mistralai").with_structured_output(Classification)
embeddings = MistralAIEmbeddings(model="mistral-embed")

#! Database Configuration
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
    
#! Define State
class State(MessagesState):
    summary: str
    user_input: str
    datas: List[Document]
    context: str
    response: str
    RAG: bool

#! Memory Configuration
# Load the conversation history from the file
# historyFile = FileChatMessageHistory("./chat_history.json")

# doit etre configuré (ou stocké, comment...) pour gardé le state
memory = MemorySaver()
config = {"configurable": {"thread_id": "abc123"}}

#! Template
prompt_template = ChatPromptTemplate.from_messages( 
    [
        (
            "system",
            "You are an AI assistant. Engage in natural, friendly conversations. "
            "Extract specific information and provide summaries when requested. "
            "Maintain a casual, helpful style"
            "Use provided documents only when the user's question directly relates to them. "
            "Return 'null' for unknown attributes or unanswerable questions. "
            "Summary of the conversation so far: {summaryContext}" 
            "the documents: {context}"
        ),
        ("human", "{text}"),
    ]
)

prompt_template_without_rag = ChatPromptTemplate.from_messages( 
    [
        (
            "system",
            "You are an AI assistant. Engage in natural, friendly conversations. "
            "Extract specific information and provide summaries when requested. "
            "Maintain a casual, helpful style"
            "Use provided documents only when the user's question directly relates to them. "
            "Return 'null' for unknown attributes or unanswerable questions. "
            "Summary of the conversation so far: {summaryContext}" 
            ),
        ("human", "{text}"),
    ]
)

summary_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant tasked with creating concise summaries. "
            "Extract only the properties mentioned in the 'Classification' class. "
            "If summaryContext is empty, summarize the last input and response. "
            "Return 'null' for unknown attributes or unanswerable questions."
            "Previous summary: {summaryContext}"
        ),
        ("human", "{text}"),
    ]
)

#! Functions Nodes
# get user input
def get_user_input(state) -> dict:
    user_input = input("Enter a prompt: ")
    return {"user_input": user_input}

def need_rag(state: State) -> bool:
    # get user input
    user_input = state["user_input"]
    
    # check if user needs RAG
    if "-use rag:" in user_input.lower():
        return {"RAG": True}
    
    return {"RAG": False}
    
# get datas from the database for making context
def retriever_similarity(state: State) -> dict:
    # transforme le prompt(query) en vecteur et fait une recherche de similarité
    retriever_similarity = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    # fait appel à l'object retriever_similarity et utlisant la méthode invoke pour obtenir les résultats
    return {"datas": retriever_similarity.invoke(state["user_input"])}

# format datas into a string
def make_context(state: State) -> dict:
    return {"context": "\n".join([doc.page_content for doc in state["datas"]])}

# generate response
# refacto il y aura pas tout le temps un context
def generate_response(state: State) -> dict:

    resume = state.get("summary", "")

    # put the context and the userPrompt in the prompt_Template
    if resume not in ["", "nothing", None]:
        if state["RAG"]:
            print("un summary et un rag")
            prompt = prompt_template.invoke({"text": state["user_input"], "context": state["context"], "summaryContext": state["summary"]})
        else:
            print("pas de rag mais un summary")
            prompt = prompt_template_without_rag.invoke({"text": state["user_input"],  "summaryContext": state["summary"]})
    else:
        if state["RAG"]: 
            print("pas de summary mais un rag")
            prompt = prompt_template.invoke({"text": state["user_input"], "context": state["context"], "summaryContext": "nothing"})
        else:
            print("pas de rag et pas de summary")
            prompt = prompt_template_without_rag.invoke({"text": state["user_input"], "summaryContext": "nothing"})

    # send prompt to the LLM
    return {"response": "\n" + llm.invoke(prompt).content + "\n"}

# get summary state, get last input and reponse messages, and make new summary with all of that
def summaryMaker(state: State) -> dict:
    # get the last input and response messages
    last_input = state["user_input"]
    last_response = state["response"]
        
    key = state.get("summary", "nothing")

    if key not in ["", "nothing", None]:
        prompt = summary_prompt_template.invoke({"text": last_input + "\n" + last_response, "summaryContext": state["summary"]})
    else:
        prompt = summary_prompt_template.invoke({"text": last_input + "\n" + last_response, "summaryContext": "nothing"})
    
    # use the llm to generate a summary and output it structured
    return {"summary": structured_llm.invoke(prompt)}

# print response
def print_response(state: State) -> None:
    print(state["response"])
    return None

# exit the program
def end_node(state: State):
    print("Exiting the program...")


#! SET UP WORKFLOW
workflow = StateGraph(state_schema=State)

# Define nodes in the workflow
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("need_rag", need_rag)
workflow.add_node("retriever_similarity", retriever_similarity)
workflow.add_node("make_context", make_context)
workflow.add_node("generate_response", generate_response)
workflow.add_node("print_response", print_response)
workflow.add_node("summaryMaker", summaryMaker)
workflow.add_node("lastNode", end_node)

# Define edges in the workflow
workflow.set_entry_point("get_user_input")

# check if the user wants to exit
workflow.add_conditional_edges("get_user_input", lambda state: state["user_input"].lower() == "exit", {True: "lastNode", False: "need_rag"})

# check if user needs RAG
workflow.add_conditional_edges("need_rag", lambda state: state["RAG"], {True: "retriever_similarity", False: "generate_response"})
# if user needs RAG
workflow.add_edge("retriever_similarity", "make_context")
workflow.add_edge("make_context", "generate_response")
# if user doesn't need RAG
workflow.add_edge("generate_response", "summaryMaker")
workflow.add_edge("summaryMaker", "print_response")

# exit Node
workflow.set_finish_point("lastNode")

# compile the workflow
app = workflow.compile(checkpointer=memory)

while True:
          
    if app.invoke({}, config)["user_input"].lower() == "exit":
       break