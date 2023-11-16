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

'command':
- "CategorySearch": Filters for recipes based on user's specified food categories or cuisines (e.g., "Italian food with chicken").
- "IngredientSearch": Find recipes using a list of ingredients provided by the user and return their titles (e.g., "prawn, spaghetti").
- "Retrieve": Search for a recipe by title and return its instructions (e.g., "Give me a recipe for Prawn Pesto Pasta").
- "Add": Add a new recipe from the user's request to the existing recipe database (e.g., "Add the following ...  to 'abalone pasta' ").
- "Remove": Remove user's specified recipe title from the existing recipe database (e.g., "Remove 'abalone pasta' from the database").
- "Modify": Modify or update existing recipe from the user's request to the existing recipe database (e.g., "update 'abalone pasta' with the following ... ").

Where relevant, return additional fields in 'recipe title' , 'recipe ingredients' , 'recipe instructions'.
- 'recipe title': Returns a string, the title of the recipe.
- 'recipe ingredients': Returns a list of string, where each element in the list is an ingredient for the recipe. If the measurements are give, remember to include it in.
- 'recipe instructions': Returns a list of string, where each element in the list is a step or instruction on how to execute the recipe.
"""
        system_prompt = system_prompt.strip('\n ')

        user_prompt_prefix  = "From the given text, identify the command. Return the json in under 'command', with optional fields 'recipe title' , 'recipe ingredients' , 'recipe instructions' where relevant"

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
        print(f"controller_inference(user_request)")
        print(f"\tuser_request = {user_request}")

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
        print(f"\t# Send request")
        print(f"\tpayload = "); pprint.pprint(payload)
        response = requests.post(self.controller_dict['openai_chat_completions_url'],
                                headers=self.controller_dict['headers'],
                                json=payload
                                )
        
        response_str = response.json()['choices'][0]['message']['content']
        response_dict = json.loads(response_str)
        print(f"\tresponse_dict = {response_dict}")

        return response_dict

    # full
    def merge_recipe_steps_ingredients(self, ingredients, instructions):
        # hardcode processing

        # ingredients
        if isinstance(ingredients, list):
            ingredients_str = "\n".join(f'- {ingredient}' for ingredient in ingredients)
        elif isinstance(ingredients, dict):
            ingredient_lst = []
            for ingredient_key, ingredient_value in ingredients.items():
                ingredient_lst.append(f"- {ingredient_key}: {ingredient_value}")
            ingredients_str = "\n".join(ingredient_lst)
        elif isinstance(ingredients, str):
            ingredients_str = ingredients
        else:
            print(ingredients)
            pdb.set_trace()

        # instructions
        if isinstance(instructions, list):
            recipe_steps_str = "\n".join(f'- {recipe_step}' for recipe_step in instructions)
        elif isinstance(instructions, dict):
            instruction_lst = []
            for instructions_key, instructions_value in instructions.items():
                instruction_lst.append(f"- {instructions_key}: {instructions_value}")
            recipe_steps_str = "\n".join(instruction_lst)
        elif isinstance(instructions, str):
            recipe_steps_str = instructions
        else:
            print(instructions)
            pdb.set_trace()

        # combined
        combined_steps = f"Ingredients:\n{ingredients_str}\n---\nInstructions:\n{recipe_steps_str}"
        return combined_steps, ingredients_str, recipe_steps_str

    def command_to_agents(self, user_request, response_dict):
        print(f"command_to_agents(user_request, response_dict)")
        command = response_dict['command']
        if command == "CategorySearch":
            _, response_lst = self.retriever_class.chat(user_request, [])
            question_text, answer_text = response_lst[0]
            return answer_text

        elif command == "IngredientSearch":
            _, all_available_recipes, _, upload_status = self.retriever_class.ingredients_to_recipes(user_request)
            return all_available_recipes

        elif command == "Retrieve":
            recipe_title_lookup, recipe_steps_lookup, ingredients_list_lookup, upload_status = \
                self.retriever_class.recipe_lookup(user_request)
            response_to_user = f"For recipe '{recipe_title_lookup}'"
            response_to_user += f"\n\n---\nIngredients:\n{ingredients_list_lookup}"
            response_to_user += f"\n\n---\nInstructions:\n{recipe_steps_lookup}"
            return response_to_user

        elif command == "Add":
            # extract recipe details
            user_recipe_title = response_dict['recipe title']
            user_recipe_ingredients = response_dict['recipe ingredients']
            user_recipe_instructions = response_dict['recipe instructions']
            _, ingredients_str, recipe_steps_str = self.merge_recipe_steps_ingredients(response_dict['recipe ingredients'],
                                                                                       response_dict['recipe instructions'])
            # add to database
            new_recipe_steps, _, upload_status = \
                self.retriever_class.add_recipe(user_recipe_title,
                                                recipe_steps_str,
                                                ingredients_str)
            return upload_status
        elif command == "Modify":
            # extract recipe details
            user_recipe_title = response_dict['recipe title']
            user_recipe_ingredients = response_dict['recipe ingredients']
            user_recipe_instructions = response_dict['recipe instructions']
            _, ingredients_str, recipe_steps_str = self.merge_recipe_steps_ingredients(response_dict['recipe ingredients'],
                                                                                       response_dict['recipe instructions'])
            # Get the closest title
            recipe_title_lookup, _recipe_response_steps, _ingredients_list, _status_message = \
                self.retriever_class.recipe_lookup(user_recipe_title)
            # Get the exact title
            new_recipe_steps, _ingredients_response, upload_status = \
                self.retriever_class.modify_recipe(recipe_title_lookup,
                                                   recipe_steps_str,
                                                   ingredients_str)
            return upload_status
        elif command == "Remove":
            user_recipe_title = response_dict['recipe title']
            # Get the closest title
            recipe_title_lookup, _, _, _ = self.retriever_class.recipe_lookup(user_recipe_title)
            # Remove this title
            _, upload_status = self.retriever_class.remove_recipe(recipe_title_lookup)
            return upload_status

    def chat_inference(self, user_request, chat_history):
        response_dict = self.controller_inference(user_request)
        response_str = self.command_to_agents(user_request, response_dict)

        chat_history.append((user_request, response_str))
        return "", chat_history

    # full
    def ocr_inference(self, image_path, chat_history):
        # perform ocr
        _, ocr_text = self.ocr_class.run_ocr(image_path)
        response_dict = self.retriever_class.ocr_to_recipe(ocr_text)

        # post process recipe
        new_recipe_title = response_dict['recipe title']
        combined_steps, _, _ = self.merge_recipe_steps_ingredients(response_dict['ingredients'],
                                                                   response_dict['recipe steps'])
        response_str = f"The identified recipe is:\n{new_recipe_title}\n---\n{combined_steps}\n"
        print(f"response_str = \n{response_str}")

        chat_history.append(("", response_str))
        return '', chat_history

def main():
    # secret keys
    load_dotenv(find_dotenv()) # read local .env file

    # setup
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

    # Load OCR model
    ocr_class = OCR()

    # Connect them
    retriever_class = Retriever(chat_bot_class,
                                embedding_class,
                                recipe_data_class,
                                )

    # re connect using controller again.
    # sorry its a mess now.
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
                                       type="filepath", # ['numpy', 'pil', 'filepath']
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
