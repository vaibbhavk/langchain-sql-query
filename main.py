from langchain import OpenAI, SQLDatabase
from langchain.chains import SQLDatabaseChain
from dotenv import load_dotenv
import os

load_dotenv()

db = SQLDatabase.from_uri(os.environ['DATABASE_URI'])
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain.from_llm(
    llm, db, verbose=True, use_query_checker=True, return_intermediate_steps=True, top_k=3)

# result = db_chain(
#     "Use the plan and tag table and return the id all the plans which have 2 bedrooms.")


result = db_chain(
    "Design a house")

# result = db_chain(
#     "Use the plan and tag table and return the id all the plans which have 2 bedrooms.")

# result = db_chain(
#     "Use the plan and tag table and return the id all the plans which have 2 bedrooms.")

print()

print("SQLResult:", result['intermediate_steps'][3])

print()

print(result['intermediate_steps'][-1].split(" "))