from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer
 
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# Create the local pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.3,
    do_sample=True
)
 
# Wrap it so LangChain can use it
llm = HuggingFacePipeline(pipeline=pipe)
 
prompt = PromptTemplate.from_template("<|user|>\nList 5 famous dishes from {country}<|assistant|>\n")
chain = prompt | llm | StrOutputParser()
 
print(chain.invoke({"country": "India"}))
 