import os
import functools
import traceback
import pdb
import pprint
import time
import requests
import json

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


class RAGcipe():
    def __init__(self,
                 ocr_class,
                 retriever_class):
        self.controller_dict = self.controller_setup()
        self.ocr_class = ocr_class
        self.retriever_class = retriever_class

    # controller
    def controller_setup(self):
        system_prompt = f"""
You are a controller agent designed to manage a recipe database.
Your role is to understand the user's request, process it based on the specified command, and return the appropriate response in JSON format.

Commands:
- "Find": Search for recipes based on user's specified food categories or cuisines (e.g., "Italian food with chicken").
- "Match": Find recipes using a list of ingredients provided by the user and return their titles (e.g., "prawn, spaghetti").
- "Scan": Upload an image of a recipe, perform OCR, and add the new recipe to the database (e.g., "help me to add this image to the existing recipe").
"""
        system_prompt = system_prompt.strip('\n ')

        user_prompt_prefix  = "From the given text, identify the command. Return in under 'command'."

        controller_dict = {
            'model_name': "gpt-3.5-turbo-1106",
            'openai_chat_completions_url': "https://api.openai.com/v1/chat/completions",
            'headers': {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            },
            'system_prompt': system_prompt,
            'user_prompt_prefix': user_prompt_prefix,
        }

        return controller_dict

    def controller_inference(self, user_request):

        user_prompt = self.controller_dict['user_prompt_prefix'] + f"\n```\n{user_request}\n```"
        user_prompt = user_prompt.strip('\n ')
        
        # Construct payload
        payload = {
            "model": self.controller_dict['model_name'],
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.controller_dict['system_prompt']},
                {"role": "user", "content": user_prompt }
            ],
            "max_tokens": 300
        }
        
        # Send request
        response = requests.post(self.controller_dict['openai_chat_completions_url'],
                                headers=self.controller_dict['headers'],
                                json=payload
                                )
        
        response_str = response.json()['choices'][0]['message']['content']
        response_dict = json.loads(response_str)

        return response_dict['command']

    def command_to_agents(self, user_request, command):
        print(f"command_to_agents(user_request, command='{command}')")
        # response_dict = {}
        if command == "Find":
            _, response_lst = self.retriever_class.chat(user_request, [])
            question_text, answer_text = response_lst[0]
            # response_dict['question'] = question_text
            # response_dict['answer'] = answer_text
            return answer_text
        elif command == "Match":
            _, all_available_recipes, _, upload_status = self.retriever_class.ingredients_to_recipes(user_request)
            # response_dict['all_available_recipes'] = all_available_recipes
            # response_dict['upload_status'] = upload_status
            return all_available_recipes
        # return response_dict

    # full
    def chat_inference(self, user_request, chat_history):
        command_str = self.controller_inference(user_request)
        response_str = self.command_to_agents(user_request, command_str)

        chat_history.append((user_request, response_str))
        return "", chat_history

    # full
    def ocr_inference(self, image_path, chat_history):
        _, ocr_text = self.ocr_class.run_ocr(image_path)
        new_recipe_title, new_recipe_steps, upload_status = self.retriever_class.ocr_to_recipe(ocr_text)

        response_str = f"The identified recipe is:\n{new_recipe_title}\n{new_recipe_steps}\n"

        chat_history.append(("", response_str))
        return '', chat_history

@debug_on_error
def main():
    # secret keys
    load_dotenv(find_dotenv()) # read local .env file

    if 1:

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

        class ChatBot():
            def __init__(self):
                self.load_chat_bot_model()

            def load_chat_bot_model(self):
                self.chat_bot = OpenAI(temperature=0)

        chat_bot_class = ChatBot()

        # Load embedding model
        embedding_class = RecipeEmbeddingsEasy(**embedding_kwargs)
        RE_CREATE_DATA = True
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

    ragcipe = RAGcipe(
        ocr_class,
        retriever_class
    )

    # Gradio Interfaces
    with gr.Blocks() as demo:
        gr.Markdown("# RAGcipe")
        with gr.Row():
            with gr.Column():
                    # chat history
                    chatbot = gr.Chatbot(height=480)
                    # user input
                    msg = gr.Textbox(label="Talk to chatbot here",
                                    value="What food are there ?")
                    # Button to send user input to the chat
                    btn = gr.Button("Submit")
                    btn.click(ragcipe.chat_inference,
                            inputs=[msg, chatbot],
                            outputs=[msg, chatbot])
                    msg.submit(ragcipe.chat_inference,
                            inputs=[msg, chatbot],
                            outputs=[msg, chatbot])
                    clear_chat = gr.ClearButton(components=[msg, chatbot],
                                                value="Clear console")

            # OCR on image
            with gr.Column():
                # Image input
                image_input = gr.Image(label="Upload Image",
                                        type="filepath", # Please choose from one of: ['numpy', 'pil', 'filepath']
                                        height=480)

                # Button to get selected recipe and upload image
                ocr_button = gr.Button("Run OCR")
                ocr_button.click(fn=ragcipe.ocr_inference,
                                    inputs=[image_input, chatbot],
                                    outputs=[msg, chatbot])

    # Gradio Launch
    demo.launch(server_port=5004,
                share=False,
                show_error=True,
                show_tips=True,
               )

if __name__ == "__main__":
    langchain.debug = True
    # langchain.debug = False
    main()

# pdb.set_trace()
