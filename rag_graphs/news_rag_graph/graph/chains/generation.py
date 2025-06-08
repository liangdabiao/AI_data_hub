from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.prompts import PromptTemplate

load_dotenv()

llm         = ChatOpenAI(model="gpt-4.1-mini",temperature=0 )
#rag_prompt  = hub.pull("rlm/rag-prompt")
rag_prompt_text =  """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=rag_prompt_text
)

generation_chain    = rag_prompt | llm | StrOutputParser()
