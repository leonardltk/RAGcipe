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

from fuzzywuzzy import fuzz
import Levenshtein
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


class RecipeData():
    def __init__(self, recipes_csv):
        self.recipes_csv = recipes_csv

        # self.allowed_cuisine_lst = [ "Seafood", "Vegetarian", "Vegan", "Western", "European", "North American", "Italian", "French", "Spanish", "Greek", "Mediterranean", "Japanese", "Chinese", "Korean", "Thai", "Vietnamese", "Indian", "Mexican", "Middle Eastern", "African", "South American", "Caribbean", "Barbecue/Grilled", "Fast Food", "Fusion", ]
        self.allowed_cuisine_lst = [ "japanese", "italian"]
        self.allowed_carbohydrates_lst = ["noodles", "rice"]
        self.allowed_proteins_lst = ["chicken", "seafood"]
        # set up the prompt config
        self.metadata_field_info, self.document_content_description = self.self_querying_retriever()

        self.recipes_df = pd.read_csv(self.recipes_csv)

        self.recipes_dict = dict(
            (row['title'], row['recipe'])
            for idx, row in self.recipes_df.iterrows()
        )

    # dataprep
    def df_to_documents(self, recipes_df):
        page_content_column = "title"
        metadata_lst = ["title", "cuisine", "Carbs", "Proteins"]
        
        split_documents = []
        for idx, row in recipes_df.iterrows():

            # create document
            current_document = Document(
                page_content = row[page_content_column],
                metadata = dict(
                    (i, row[i])
                    for i in metadata_lst
                ),
            )

            # append to document list
            split_documents.append(current_document)
        return split_documents

    def data_prep(self):

        split_documents = self.df_to_documents(self.recipes_df)

        return split_documents

    # recipe lookup
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
                description=f"The type of cuisine that the recipe makes. Choose from one of {self.allowed_cuisine_lst}",
                type="string",
            ),
            AttributeInfo(
                name="Carbs",
                description=f"The kind of carbohydrates used as the base for the recipe. Choose from one of {self.allowed_carbohydrates_lst}",
                type="string",
            ),
            AttributeInfo(
                name="Proteins",
                description=f"The proteins or meats used for the recipe. Choose from one of {self.allowed_proteins_lst}",
                type="string",
            ),
        ]
        document_content_description = "Brief descriptions of the recipe"
        return metadata_field_info, document_content_description

    # add/remove/modify recipes
    def add_data(self, add_recipes_df):
        print("\tadd_data()")
        recipe_title = add_recipes_df['title'][0]
        recipe_steps = add_recipes_df['recipe'][0]
        
        split_documents = self.df_to_documents(add_recipes_df)

        # Add to DataFrame
        self.recipes_df = pd.concat([self.recipes_df, add_recipes_df])
        self.recipes_df.reset_index(drop=True, inplace=True)
        print(f"\t\tSucessfully added {recipe_title} to self.recipes_df")

        # Add to dictionary
        self.recipes_dict[recipe_title] = recipe_steps
        print(f"\t\tSucessfully added {recipe_title} to self.recipes_dict")

        # Add to csv
        self.recipes_df.to_csv(self.recipes_csv, index=False)
        print(f"\t\tWritten to self.recipes_csv={self.recipes_csv}")

        return split_documents

    def modify_data(self, recipe_title, recipe_steps):
        print("\tmodify_data(recipe_title, recipe_steps)")
        status_fail_bool = None
        status_message = None

        # Check existence
        if not recipe_title in self.recipes_dict:
            existing_title = ""
            potential_score = -1
            for existing_title in self.recipes_dict:
                text_similarity_score = Levenshtein.distance(recipe_title, existing_title)
                text_similarity_score = 1 - text_similarity_score/max(len(recipe_title), len(existing_title))
                if (text_similarity_score > potential_score) and (text_similarity_score > 0.8):
                    potential_score = text_similarity_score
                    potential_title = existing_title
            if potential_score > 0.8:
                status_message = f"{recipe_title} don't exists, did you mean {potential_title} ?"
            else:
                status_message = f"{recipe_title} don't exists."
            status_fail_bool = True
        else:
            status_fail_bool = False
            status_message = f"{recipe_title} exists."

        # Modify DataFrame
        UPDATED = False
        for idx, row in self.recipes_df.iterrows():
            if row['title'] == recipe_title:
                self.recipes_df['recipe'][idx] = recipe_steps
                print(f'\t\t\trow["recipe"] <- recipe_steps={recipe_steps} | row["recipe"] = {row["recipe"]}')
                UPDATED = True
                if self.recipes_df['recipe'][idx] != recipe_steps:
                    pdb.set_trace()
        if UPDATED == False:
            pdb.set_trace()
        print(f'\t\tUpdated {recipe_title} to self.recipes_df')

        # Modify dictionary
        self.recipes_dict[recipe_title] = recipe_steps
        print(f'\t\tUpdated {recipe_title} to self.recipes_dict')

        # Modify csv
        self.recipes_df.to_csv(self.recipes_csv, index=False)
        print(f"\t\tWritten to self.recipes_csv={self.recipes_csv}")

    def remove_data(self, recipe_title):
        print("\tremove_data()")

        self.recipes_df.drop(
            self.recipes_df[self.recipes_df['title'] == recipe_title].index,
            inplace = True
        )
        self.recipes_df.reset_index(drop=True, inplace=True)
        print(f"\t\tSucessfully removed {recipe_title} from self.recipes_df")

        recipe_steps = self.recipes_dict.pop(recipe_title)
        print(f"\t\tSucessfully removed {recipe_title} from self.recipes_dict")

        self.recipes_df.to_csv(self.recipes_csv, index=False)
        print(f"\t\tWritten to self.recipes_csv={self.recipes_csv}")


class Retriever():
    def __init__(self,
                 chat_bot_class,
                 embedding_class,
                 data_class,
                ):

        self.chat_bot_class = chat_bot_class
        self.embedding_class = embedding_class
        self.data_class = data_class

        self.retriever = SelfQueryRetriever.from_llm(
            llm = self.chat_bot_class.chat_bot,
            vectorstore = self.embedding_class.vector_db,
            document_contents = self.data_class.document_content_description,
            metadata_field_info = self.data_class.metadata_field_info,
            verbose = True
        )

    # chat lookup
    def chat(self, user_input, past_history):
        # Ask question for retriever to self-query
        # user_input = "what are some dishes that has either seafood or chicken proteins ?"
        output_lst = self.retriever.get_relevant_documents(user_input)
        print(f"output_lst = {output_lst}")

        # Process the response
        if len(output_lst) == 0:
            LLM_response = "No recipe found based on what you're asking for, please try again."
        else:
            response_lst = []
            for idx, output in enumerate(output_lst):
                response_lst.append(f"{idx}. {output.page_content}")
            LLM_response = "\n".join(response_lst)

        # Add to chat history
        past_history.append((user_input, LLM_response))

        return "", past_history

    def lookup_recipe(self, user_input, past_history):
        output_lst = self.retriever.get_relevant_documents(user_input)
        response_lst = []
        for idx, output in enumerate(output_lst):
            response_lst.append(f"{idx}. {output.page_content}")
        LLM_response = "\n".join(response_lst)

        past_history.append((user_input, LLM_response))

        return "", past_history

    # embedding lookup
    def recipe_lookup(self, standalone_question):
        recipe_response_title = "No recipe found"
        recipe_response_steps = ""

        recipes_dict = self.data_class.recipes_dict
        documents = self.embedding_class.get_documents(standalone_question)

        # get context as string
        for idx, document in enumerate(documents):
            print(f"{idx}: {document.page_content}")
            if idx == 0:
                recipe_response_title = document.metadata['title']
                recipe_response_steps = recipes_dict.get(recipe_response_title, f"'{recipe_response_title}' not in recipes_dict, please check.")

        return recipe_response_title, recipe_response_steps

    # add/remove/modify recipes
    def add_recipe(self, recipe_title, recipe_steps):
        # form recipe df
        add_recipes_df = pd.DataFrame({
            'title': [recipe_title],
            'recipe': [recipe_steps],
            'cuisine': ['cuisine'],
            'Carbs': ['Carbs'],
            'Proteins': ['Proteins'],
        })

        # Add to data_class
        split_documents = self.data_class.add_data(add_recipes_df)

        # Add to embedding_class
        self.embedding_class.add_vector_db(split_documents)

        # ------ Sanity check ------
        try:
            # --- data_class ---
            # recipes_csv, recipes_df
            for idx, tmp_df in enumerate([
                    pd.read_csv(self.data_class.recipes_csv),
                    self.data_class.recipes_df.copy()
                ]):
                print(f'idx={idx}')
                # tmp_df
                index_list = list(tmp_df.index)
                assert len(index_list) == len(set(index_list))
                # this_recipe_df
                this_recipe_df =  tmp_df[tmp_df['title']==recipe_title]
                assert len(this_recipe_df) == 1
                for idx, row in this_recipe_df.iterrows():
                    assert row['recipe'] == recipe_steps
            # recipes_dict
            assert recipe_steps == self.data_class.recipes_dict.get(recipe_title, '')

            # --- embedding_class ---
            # Add to vector_db
            tmp_dict = self.embedding_class.vector_db.get()
            # documents
            document_lst = tmp_dict['documents']
            assert len(document_lst) == len(set(document_lst))
            assert recipe_title in document_lst
            # metadatas
        except:
            traceback.print_exc()
            pdb.set_trace()

        return f"{recipe_title} successfully added"

    def remove_recipe(self, recipe_title):

        # Add to embedding_class
        self.embedding_class.remove_vector_db(recipe_title)

        # Add to data_class
        self.data_class.remove_data(recipe_title)

        # ------ Sanity check ------
        try:
            # --- data_class ---
            # recipes_csv, recipes_df
            for idx, tmp_df in enumerate([
                    pd.read_csv(self.data_class.recipes_csv),
                    self.data_class.recipes_df.copy()
                ]):
                print(f'idx={idx}')
                # tmp_df
                index_list = list(tmp_df.index)
                assert len(index_list) == len(set(index_list))
                # this_recipe_df
                this_recipe_df =  tmp_df[tmp_df['title']==recipe_title]
                assert len(this_recipe_df) == 0
            # recipes_dict
            assert '' == self.data_class.recipes_dict.get(recipe_title, '')

            # --- embedding_class ---
            # Add to vector_db
            tmp_dict = self.embedding_class.vector_db.get()
            # documents
            document_lst = tmp_dict['documents']
            assert len(document_lst) == len(set(document_lst))
            assert not recipe_title in document_lst
            # metadatas
        except:
            traceback.print_exc()
            pdb.set_trace()

        return f"{recipe_title} successfully removed"

    def modify_recipe(self, recipe_title, recipe_steps):
        print(f'modify_recipe()')

        # Update data_class
        self.data_class.modify_data(recipe_title, recipe_steps)

        # Update embedding_class
        pass
 
        # ------ Sanity check ------
        try:
            # --- data_class ---
            # recipes_csv, recipes_df
            for idx, tmp_df in enumerate([
                    pd.read_csv(self.data_class.recipes_csv),
                    self.data_class.recipes_df.copy(),
                ]):
                print(f'idx={idx}')
                # tmp_df
                index_list = list(tmp_df.index)
                assert len(index_list) == len(set(index_list))
                # this_recipe_df
                this_recipe_df =  tmp_df[tmp_df['title']==recipe_title]
                assert len(this_recipe_df) == 1
                for idx, row in this_recipe_df.iterrows():
                    assert row['recipe'] == recipe_steps

                    """
                        tmp_df  = self.data_class.recipes_df.copy()
                    """

                    """
                        # Modify DataFrame
                        for idx, row in self.recipes_df.iterrows():
                            if row['title'] == recipe_title:
                                row['recipe'] = recipe_steps
                        print(f'\t\tUpdated {recipe_title} to self.recipes_dict')
                    """

            # recipes_dict
            assert recipe_steps == self.data_class.recipes_dict.get(recipe_title, '')

            # --- embedding_class ---
            # Add to vector_db
            tmp_dict = self.embedding_class.vector_db.get()
            # documents
            document_lst = tmp_dict['documents']
            assert len(document_lst) == len(set(document_lst))
            assert recipe_title in document_lst
            # metadatas
        except:
            traceback.print_exc()
            pdb.set_trace()

        return f"{recipe_title} successfully modified"

    def list_recipe(self):
        print(f'list_recipe()')

        all_titles = sorted(self.data_class.recipes_dict)
        all_titles = [ f'{idx}. {title}' for idx, title in enumerate(all_titles, 1)]
        all_titles = "\n".join(all_titles)

        upload_status = "Success: Listed all recipes from database"

        return all_titles, upload_status


RE_CREATE_DATA = True
RE_CREATE_DATA = False

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
