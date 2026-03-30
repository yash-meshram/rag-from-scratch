from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from typing import List, Dict, Any

class GroqLLM:
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        api_key: str = None
    ):
        '''
        Initializing Groq LLM
        '''
        load_dotenv()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("\nGroq API key is not specified. Set GROQ_API_KEY environment variable or pass api_key parameter")
        
        self.llm = ChatGroq(
            model = self.model_name,
            api_key = self.api_key,
            temperature = 0.1,
            max_tokens = 1024
        )
        print(f"Iniializing Groq LLM with model: {self.model_name}")
        
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> dict:
        '''
        Generate the response
        prompt = query + retrieved_docs (retrieved docs text from vector store)
        prompt -> llm -> response
        '''
        # create prompt template
        prompt = PromptTemplate(
            input_variables = ["query", "referance"],
            template = 
            """
            # Question: {query}
            # Referance: {referance}
            # Instruction: 
            1. Use the given 'Content' only to generate the answer of the 'Question'.
            2. Use only the information provided to you in 'Content'. Dont use any other information even if it was asked to.
            3. Provide the Answer to the 'Question' only, do not provide any preamble.
            """
        )
        
        chain = prompt | self.llm
        
        response = chain.invoke(
            input = {
                "query": query,
                "referance": [doc["content"] for doc in retrieved_docs]
            }
        )
        
        output = {
            'content': response.content,
            'source': list({doc["metadata"]["file_name"] for doc in retrieved_docs})
        }
        
        return output