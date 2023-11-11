import os
import functools
import traceback
import pdb
import pprint
import time

import gradio as gr
from dotenv import load_dotenv, find_dotenv
from termcolor import colored

import langchain
from langchain.llms import OpenAI
# from openai import OpenAI

from recipe_class_data import RecipeData
from recipe_class_embedding import RecipeEmbeddingsEasy
from recipe_class_retriever import Retriever
from recipe_class_ocr import OCR

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

        # completions = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello!"}])
        # print(completions.choices[0].message)


        # pdb.set_trace()



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
        titles_documents, ingredients_documents = recipe_data_class.data_prep()
        embedding_class.create_vector_db(titles_documents, ingredients_documents)

        # debug
        for standalone_question in ['chinese food', 'italian food']:
            documents = embedding_class.get_documents(standalone_question, vector_db_name='titles_db')
            print(standalone_question)
            for document in documents:
                print(f'\t{document.page_content}')
    else:
        vector_db_name_lst = list(embedding_class.vector_dbs)
        for vector_db_name in vector_db_name_lst:
            embedding_class.read_vector_db(vector_db_name=vector_db_name)

    # Load OCR model
    ocr_class = OCR()

    # Connect them
    retriever_class = Retriever(chat_bot_class,
                                embedding_class,
                                recipe_data_class,
                                )

    # Gradio Interfaces
    with gr.Blocks() as demo:
        gr.Markdown("# RAGcipe")
        with gr.Row():
            # Chatbot
            with gr.Column():
                # chat history
                chatbot = gr.Chatbot(height=480)
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
                clear_chat = gr.ClearButton(components=[msg, chatbot],
                                            value="Clear console")

            # Updating Recipes
            with gr.Column():
                # Recipe Title
                with gr.Row():
                    # Textbox to type recipe
                    new_recipe_title = gr.Textbox(label="Recipe Title",
                                                  lines=1)
                # Steps/Titles, Ingredients
                with gr.Row():
                    line_number = 10
                    with gr.Column():
                        new_recipe_steps = gr.Textbox(label="""Steps / Titles""",
                                                      lines=line_number)
                    with gr.Column():
                        # Textbox to display/read ingredients
                        ingredients_list = gr.Textbox(label="Ingredients list",
                                                      lines=line_number)
                # Upload Status
                with gr.Row():
                        # Textbox to display upload status
                        upload_status = gr.Textbox(label="Upload Status",
                                                   lines=1)

                # Interact buttons
                with gr.Row():
                    # Add / Remove / Modify
                    with gr.Column():
                        # Button to add new recipe
                        recipe_add = gr.Button("Add recipe")
                        recipe_add.click(retriever_class.add_recipe,
                                        inputs=[new_recipe_title, new_recipe_steps],
                                        outputs=[new_recipe_steps, ingredients_list, upload_status])
                        # Button to remove recipe
                        recipe_remove = gr.Button("Remove recipe")
                        recipe_remove.click(retriever_class.remove_recipe,
                                            inputs=[new_recipe_title],
                                            outputs=[new_recipe_steps, upload_status])
                        # Button to add new recipe
                        recipe_modify = gr.Button("Modify recipe")
                        recipe_modify.click(retriever_class.modify_recipe,
                                            inputs=[new_recipe_title, new_recipe_steps],
                                            outputs=[new_recipe_steps, ingredients_list, upload_status])
                    # Extraction
                    with gr.Column():
                        # Button to ask for recipe
                        btn_recipe = gr.Button("Title -> Steps")
                        btn_recipe.click(retriever_class.recipe_lookup,
                                        inputs=[new_recipe_title],
                                        outputs=[new_recipe_title, new_recipe_steps, ingredients_list, upload_status])

                        # Button to list all recipes
                        recipe_list = gr.Button("List all recipes")
                        recipe_list.click(retriever_class.list_recipe,
                                        inputs=[],
                                        outputs=[new_recipe_steps, upload_status])

                        # Button to show list of recipes with ingredients
                        btn_recipe = gr.Button("Ingredients -> Recipes")
                        btn_recipe.click(retriever_class.ingredients_to_recipes,
                                        inputs=[ingredients_list],
                                        outputs=[new_recipe_title, new_recipe_steps, ingredients_list, upload_status])

                # Misc
                with gr.Row():
                    with gr.Column():
                        # clear console
                        clear_recipe = gr.ClearButton(components=[new_recipe_title, new_recipe_steps, ingredients_list, upload_status],
                                                      value="Clear console")
                        # pdb
                        btn_recipe = gr.Button("pdb")
                        btn_recipe.click(retriever_class.pdb,
                                        inputs=[new_recipe_title, new_recipe_steps, ingredients_list],
                                        outputs=[new_recipe_title, new_recipe_steps, ingredients_list, upload_status])

            # OCR on image
            with gr.Column():
                # Image input
                image_input = gr.Image(label="Upload Image",
                                       type="filepath", # Please choose from one of: ['numpy', 'pil', 'filepath']
                                       height=480)

                # Textbox to display selected recipe
                ocr_text = gr.Textbox(label="OCR texts",
                                      lines=1)

                # Button to get selected recipe and upload image
                ocr_button = gr.Button("Run OCR")
                ocr_button.click(fn=ocr_class.run_ocr,
                                 inputs=[image_input],
                                 outputs=[image_input, ocr_text])

                # Button to identify recipe
                ocr_to_recipe_button = gr.Button("Identify recipe")
                ocr_to_recipe_button.click(fn=retriever_class.ocr_to_recipe,
                                 inputs=[ocr_text],
                                 outputs=[new_recipe_title, new_recipe_steps, upload_status])

    # Gradio Launch
    demo.launch(server_port=5004,
                share=False,
                show_error=True,
                show_tips=True,
               )

if __name__ == "__main__":
    langchain.debug = True
    langchain.debug = False
    main()

# pdb.set_trace()
