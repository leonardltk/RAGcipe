import openai

import pdb
import functools
import traceback, pdb, pprint
import os
import io
import time
import base64
import argparse
import re

import pandas as pd
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from termcolor import colored

import openai

import langchain
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from embedding_recipe import RecipeEmbeddingsEasy


class ChatBot():
    def __init__(self):
        self.load_chat_bot_model()

    def load_chat_bot_model(self):
        self.chat_bot = OpenAI(temperature=0)


class RecipeData():
    def __init__(self, recipes_csv):
        self.recipes_csv = recipes_csv
        self.metadata_field_info, self.document_content_description = self.self_querying_retriever()

    def data_prep(self, recipes_csv):
        recipes_df = pd.read_csv(recipes_csv)

        page_content_column = "title"
        metadata_lst = ["title", "cuisine", "Carbs", "Proteins"]

        split_documents = []
        for idx, row in recipes_df.iterrows():
            current_document = Document(
                page_content = row[page_content_column],
                metadata = dict(
                    (i, row[i])
                    for i in metadata_lst
                ),
            )

            split_documents.append(current_document)

        return recipes_df, split_documents

    def self_querying_retriever(self):
        # Creating our self-querying retriever
        metadata_field_info = [
            AttributeInfo(
                name="title",
                description="Title of the recipe",
                type="string",
            ),
            AttributeInfo(
                name="cuisine",
                description="The type of cuisine that the recipe makes",
                type="string or list[string]",
            ),
            AttributeInfo(
                name="Carbs",
                description="The kind of carbohydrates used as the base for the recipe",
                type="string or list[string]",
            ),
            AttributeInfo(
                name="Proteins",
                description="The proteins or meats used for the recipe",
                type="string or list[string]",
            ),
        ]
        document_content_description = "Brief descriptions of the recipe"
        return metadata_field_info, document_content_description


class Retriever():
    def __init__(self,
                 chat_bot_class,
                 embedding_class,
                 data_class,
                ):
        self.retriever = SelfQueryRetriever.from_llm(
            chat_bot_class.chat_bot,
            embedding_class.vector_db,
            data_class.document_content_description,
            data_class.metadata_field_info,
            verbose=True
        )

    def lookup_recipes(self, user_input, past_history):
        output_lst = self.retriever.get_relevant_documents(user_input)
        response_lst = []
        for idx, output in enumerate(output_lst):
            response_lst.append(f"{idx}. {output.page_content}")
        LLM_response = "\n".join(response_lst)

        past_history.append((user_input, LLM_response))

        return "", past_history

HAVE_CREATED_DATA = True
# HAVE_CREATED_DATA = False

def main():
    # secret keys
    load_dotenv(find_dotenv()) # read local .env file

    # Init
    recipes_csv = 'data/recipes.csv'
    embedding_kwargs = {
        "embedding_model_name" : "hkunlp/instructor-base", # "hkunlp/instructor-xl"
        "CHUNKS_TXT" : "docs/chunks.txt",
        "CHROMA_DIR" : "docs/chroma/",
        "RETRIEVER_KWARGS" : {
            "search_type": "similarity", # {similarity, similarity_score_threshold, mmr}
            "samples": 5
        }
    }

    # Dataprep
    recipe_data_class = RecipeData(recipes_csv)

    # Load Chatbot model
    chat_bot_class = ChatBot()

    # Load embedding model
    embedding_class = RecipeEmbeddingsEasy(**embedding_kwargs)
    if HAVE_CREATED_DATA:
        embedding_class.read_vector_db()
    else:
        recipes_df, split_documents = recipe_data_class.data_prep(recipes_csv)
        metadata_field_info, document_content_description = recipe_data_class.self_querying_retriever()
        embedding_class.create_vector_db(split_documents)

        # debug
        standalone_question = 'chinese food'
        documents = embedding_class.get_context(standalone_question)
        print(standalone_question, documents)

        standalone_question = 'italian food'
        documents = embedding_class.get_context(standalone_question)
        print(standalone_question, documents)

    # Connect them
    retriever_class = Retriever(chat_bot_class,
                                embedding_class,
                                recipe_data_class,
                                )

    # Gradio Interfaces
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(height=480) # just to fit the notebook
        msg = gr.Textbox(label="Prompt")
        btn = gr.Button("Submit")
        clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
        btn.click(retriever_class.lookup_recipes, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(retriever_class.lookup_recipes, inputs=[msg, chatbot], outputs=[msg, chatbot]) # Press enter to submit

    # Gradio Launch
    demo.launch(server_port=5004,
                share=False,
                show_error=True,
                show_tips=True,
               )

if __name__ == "__main__":
    langchain.debug = True
    main()
