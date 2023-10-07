import os
import functools
import traceback
import pdb
import pprint
import time
# import base64
# import argparse

# from fuzzywuzzy import fuzz
# import Levenshtein
# import pandas as pd
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from termcolor import colored

# import openai

import langchain
# from langchain.schema import Document
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma

from langchain.llms import OpenAI
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.chains.query_constructor.base import AttributeInfo

from recipe_class_data import RecipeData
from recipe_class_embedding import RecipeEmbeddingsEasy
from recipe_class_retriever import Retriever

def debug_on_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n----- Exception occurred: {e} ----- ")
            traceback.print_exc()
            print(f"----------------------------------------")
            pdb.post_mortem()
    return wrapper


class ChatBot():
    def __init__(self):
        self.load_chat_bot_model()

    def load_chat_bot_model(self):
        self.chat_bot = OpenAI(temperature=0)



RE_CREATE_DATA = True
# RE_CREATE_DATA = False

@debug_on_error
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
    if RE_CREATE_DATA:
        split_documents = recipe_data_class.data_prep()
        metadata_field_info, document_content_description = recipe_data_class.self_querying_retriever()
        embedding_class.create_vector_db(split_documents)

        # debug
        standalone_question = 'chinese food'
        documents, context_string = embedding_class._get_context(standalone_question)
        print(standalone_question)
        for document in documents:
            print(f'\t{document.page_content}')

        standalone_question = 'italian food'
        documents, context_string = embedding_class._get_context(standalone_question)
        print(standalone_question)
        for document in documents:
            print(f'\t{document.page_content}')
    else:
        embedding_class.read_vector_db()

    # Connect them
    retriever_class = Retriever(chat_bot_class,
                                embedding_class,
                                recipe_data_class,
                                )

    # Gradio Interfaces
    version = "v2"
    with gr.Blocks() as demo:
        gr.Markdown("# RAGcipe")
        with gr.Row():
            # Chatbot
            with gr.Column():
                # chat history
                chatbot = gr.Chatbot(height=480)
                clear = gr.ClearButton(components=[chatbot],
                                       value="Clear console")
                # user input
                msg = gr.Textbox(label="Talk to chatbot here",
                                 value="What food are there ?")
                # Button to send user input to the cchat
                btn = gr.Button("Submit")
                btn.click(retriever_class.chat,
                          inputs=[msg, chatbot],
                          outputs=[msg, chatbot])
                msg.submit(retriever_class.chat,
                           inputs=[msg, chatbot],
                           outputs=[msg, chatbot])

            # Updating Recipes
            with gr.Column():
                # Textbox to type recipe
                new_recipe_title = gr.Textbox(label="Recipe Title", lines=1)
                new_recipe_steps = gr.Textbox(label="{Add,Modify}:Type steps here\nList:Show recipe titles",
                                              lines=10)

                # Textbox to display upload status
                upload_status = gr.Textbox(label="Upload Status", lines=1)

                # Button to add new recipe
                recipe_add = gr.Button("Add")
                recipe_add.click(retriever_class.add_recipe,
                                 inputs=[new_recipe_title, new_recipe_steps],
                                 outputs=[upload_status])
                # Button to remove recipe
                recipe_remove = gr.Button("Remove")
                recipe_remove.click(retriever_class.remove_recipe,
                                    inputs=[new_recipe_title],
                                    outputs=[upload_status])
                # Button to add new recipe
                recipe_modify = gr.Button("Modify")
                recipe_modify.click(retriever_class.modify_recipe,
                                    inputs=[new_recipe_title, new_recipe_steps],
                                    outputs=[upload_status])
                # Button to list all recipes
                recipe_list = gr.Button("List")
                recipe_list.click(retriever_class.list_recipe,
                                  inputs=[],
                                  outputs=[new_recipe_steps, upload_status])

                # Chat to ask for recipe
                recipe_request = gr.Textbox(label="Ask for recipe here",
                                            value="What is the recipe for pasta ?",
                                            lines=1)
                # Button to ask for recipe
                btn_recipe = gr.Button("Ask for recipe")
                btn_recipe.click(retriever_class.recipe_lookup,
                                 inputs=[recipe_request],
                                 outputs=[new_recipe_title, new_recipe_steps])
                recipe_request.submit(retriever_class.recipe_lookup,
                                      inputs=[recipe_request],
                                      outputs=[new_recipe_title, new_recipe_steps])  # Press enter to submit

    # Gradio Launch
    demo.launch(server_port=5004,
                share=False,
                show_error=True,
                show_tips=True,
               )

if __name__ == "__main__":
    langchain.debug = True
    main()

# pdb.set_trace()
