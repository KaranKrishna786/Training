# Scenario: AI-Powered Quiz System with LangChain
# Objective
# Assess learners’ understanding of a topic 
# (e.g., Python basics, machine learning concepts, or workplace compliance) 
# by dynamically generating questions using an open-source LLM integrated with LangChain.

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer
 
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
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
 
prompt = PromptTemplate.from_template("<|user|>\nList 10 top question of {course}<|assistant|>\n")
chain = prompt | llm | StrOutputParser()
 
print(chain.invoke({"course": "Python"}))
 