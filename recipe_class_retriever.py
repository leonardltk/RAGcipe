import pprint
import pdb
import traceback
import json
import re

import pandas as pd

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

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
            vectorstore = self.embedding_class.vector_dbs['titles_db'],
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

    # embedding lookup
    def recipe_lookup(self, standalone_question):
        recipe_response_title = "No recipe found"
        recipe_response_steps = ""
        status_message = ""
        SIMILAR_RECIPE_MSG = "\nSimilar recipes:"

        recipes_dict = self.data_class.recipes_dict
        documents = self.embedding_class.get_documents(standalone_question,
                                                       vector_db_name="titles_db")

        # get context as string
        for idx, document in enumerate(documents):
            print(f"{idx}: {document.page_content}")
            if idx == 0:
                recipe_response_title = document.metadata['title']
                if recipe_response_title in recipes_dict:
                    recipe_response_steps = recipes_dict[recipe_response_title]
                    if standalone_question != recipe_response_title:
                        status_message = f"You asked for '{standalone_question}', but we returned '{recipe_response_title}'."
                    else:
                        status_message = f"'{standalone_question}' successfully retrieved."
                    status_message += SIMILAR_RECIPE_MSG
                    
                    extracted_df = self.data_class.recipes_df[self.data_class.recipes_df['title'] == recipe_response_title]
                    for idx, row in extracted_df.iterrows():
                        ingredients_list = row['ingredients']

                else:
                    recipe_response_steps = ""
                    status_message = f"'{recipe_response_title}' not in recipes_dict, please check."
            else:
                recipe_response_title_alternatives = document.metadata['title']
                status_message += f"\n\t{idx}. {recipe_response_title_alternatives}"
        if status_message.endswith(SIMILAR_RECIPE_MSG):
            status_message.replace(SIMILAR_RECIPE_MSG, "")


        return recipe_response_title, recipe_response_steps, ingredients_list, status_message

    # LLM extract info from recipe
    def get_categories_from_recipe(self, recipe_title, recipe_steps):
        prompt_to_submit = self.data_class.categories_extraction_prompt_template.format(
            recipe_title=recipe_title,
            recipe_steps=recipe_steps
        )
        # inference
        categories_response = self.chat_bot_class.chat_bot(prompt_to_submit)
        categories_response = re.sub(r'\{+', '{', categories_response)
        categories_response = re.sub(r'\}+', '}', categories_response)
        print(f"\n\tcategories_response = {categories_response}", end='\n---\n')
        categories_json = json.loads(categories_response)
        print(f"\n\tcategories_json = {categories_json}", end='\n---\n')
        return categories_json

    def get_ingredients_from_recipe(self, recipe_title, recipe_steps):
        prompt_to_submit = self.data_class.ingredients_extraction_prompt_template.format(
            recipe_title=recipe_title,
            recipe_steps=recipe_steps
        )
        # inference
        ingredients_response = self.chat_bot_class.chat_bot(prompt_to_submit)
        ingredients_response = re.sub(r'\{+', '{', ingredients_response)
        ingredients_response = re.sub(r'\}+', '}', ingredients_response)
        ingredients_response = ingredients_response.strip('\n')
        print(f"\n\tingredients_response = {ingredients_response}", end='\n---\n')
        return ingredients_response

    # add/remove/modify recipes
    def add_recipe(self, recipe_title, recipe_steps):
        print(f'add_recipe()')
        # LLM to generate the cuisine / carbs / proteins
        categories_json = self.get_categories_from_recipe(recipe_title, recipe_steps)

        # LLM to generate the ingredients
        ingredients_response = self.get_ingredients_from_recipe(recipe_title, recipe_steps)

        # form recipe df
        add_recipes_df = pd.DataFrame({
            'title': [recipe_title],
            'recipe': [recipe_steps],
            'cuisine': [categories_json['Cuisines']],
            'Carbs': [categories_json['Carbohydrates']],
            'Proteins': [categories_json['Proteins']],
            'ingredients': ingredients_response,
        })

        # Add to data_class
        titles_documents, ingredients_documents = self.data_class.add_data(add_recipes_df)

        # Add to embedding_class
        self.embedding_class.add_vector_db(titles_documents,
                                           vector_db_name="titles_db")
        self.embedding_class.add_vector_db(ingredients_documents,
                                           vector_db_name="ingredients_db")

        # ------ Sanity check ------
        try:
            kwargs_dict={'recipe_title':recipe_title, 'recipe_steps':recipe_steps}
            self.data_class.sanity_check(mode="add", kwargs_dict=kwargs_dict)
            self.embedding_class.sanity_check(mode="add", kwargs_dict=kwargs_dict)
        except:
            traceback.print_exc()
            pdb.set_trace()

        return recipe_steps, ingredients_response, f"{recipe_title} successfully added"

    def modify_recipe(self, recipe_title, recipe_steps):
        print(f'modify_recipe()')
        # LLM to generate the cuisine / carbs / proteins
        categories_json = self.get_categories_from_recipe(recipe_title, recipe_steps)

        # LLM to generate the ingredients
        ingredients_response = self.get_ingredients_from_recipe(recipe_title, recipe_steps)

        # form recipe df
        modify_recipes_df = pd.DataFrame({
            'title': [recipe_title],
            'recipe': [recipe_steps],
            'cuisine': [categories_json['Cuisines']],
            'Carbs': [categories_json['Carbohydrates']],
            'Proteins': [categories_json['Proteins']],
            'ingredients': ingredients_response,
        })

        # Update data_class
        titles_documents, ingredients_documents = self.data_class.modify_data(modify_recipes_df)

        # Update embedding_class
        self.embedding_class.modify_vector_db(titles_documents,
                                              vector_db_name="titles_db")
        self.embedding_class.modify_vector_db(ingredients_documents,
                                              vector_db_name="ingredients_db")
 
        # ------ Sanity check ------
        try:
            kwargs_dict={'recipe_title':recipe_title, 'recipe_steps':recipe_steps}
            self.data_class.sanity_check(mode="modify", kwargs_dict=kwargs_dict)
            self.embedding_class.sanity_check(mode="modify", kwargs_dict=kwargs_dict)
        except:
            traceback.print_exc()
            pdb.set_trace()

        return recipe_steps, ingredients_response, f"{recipe_title} successfully modified"

    def remove_recipe(self, recipe_title):
        print(f'remove_recipe()')
        # Add to embedding_class
        self.embedding_class.remove_vector_db(recipe_title,
                                              vector_db_name="titles_db")
        self.embedding_class.remove_vector_db(recipe_title,
                                              vector_db_name="ingredients_db")

        # Add to data_class
        self.data_class.remove_data(recipe_title)

        # ------ Sanity check ------
        try:
            kwargs_dict={'recipe_title':recipe_title}
            self.data_class.sanity_check(mode="remove", kwargs_dict=kwargs_dict)
            self.embedding_class.sanity_check(mode="remove", kwargs_dict=kwargs_dict)
        except:
            traceback.print_exc()
            pdb.set_trace()

        return "", f"{recipe_title} successfully removed"

    def list_recipe(self):
        print(f'list_recipe()')

        all_titles = sorted(self.data_class.recipes_dict)
        all_titles = [ f'{idx}. {title}' for idx, title in enumerate(all_titles, 1)]
        all_titles = "\n".join(all_titles)

        upload_status = "Success: Listed all recipes from database"

        return all_titles, upload_status

    # ingredients_to_recipes
    def ingredients_to_recipes(self, ingredients_string):
        recipe_string = ""

        # # ================ BM25Retriever search ================
        # bm25_retriever = BM25Retriever.from_texts(self.data_class.recipes_df['ingredients'])
        # document_lst = bm25_retriever.get_relevant_documents(ingredients_string)

        # ================ semenatic search ================
        document_lst = self.embedding_class.get_documents(ingredients_string,
                                                        vector_db_name="ingredients_db")
        for idx, document in enumerate(document_lst, 1):
            recipe_string += f"{idx}. {document.metadata['title']}\n"

        recipe_string = recipe_string.strip('\n')

        return "", recipe_string, ingredients_string, f"recipes successfully retrieved"

    # pdb
    def pdb(self, new_recipe_title, new_recipe_steps, ingredients_string):
        upload_status = ""
        pdb.set_trace()
        return new_recipe_title, new_recipe_steps, ingredients_string, upload_status
