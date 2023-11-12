import os
import pprint
import pdb
import traceback
import json
import re
import requests

import openai
import pandas as pd

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from langchain.vectorstores import FAISS

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
            for idx, output in enumerate(output_lst, 1):
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

    def get_title(self, document):
        ingredient = document.page_content
        title_bool = self.data_class.recipes_df['ingredients'] == ingredient
        for extracted_title in self.data_class.recipes_df[title_bool]['title']:
            pass
        return extracted_title

    # ingredients_to_recipes
    def ingredients_to_recipes(self, ingredients_string, display_titles=3):
        # ================ BM25Retriever search ================
        bm25_retriever = BM25Retriever.from_texts(self.data_class.recipes_df['ingredients'])
        document_lst = bm25_retriever.get_relevant_documents(ingredients_string)
        recipe_string_bm25 = ""
        for idx, document in enumerate(document_lst, 1):
            extracted_title = self.get_title(document)
            recipe_string_bm25 += f"{idx}. {extracted_title}\n"
            if idx >= display_titles:
                break
        recipe_string_bm25 = recipe_string_bm25.strip('\n')
        print("--- titles (bm25) ---")
        print(recipe_string_bm25)

        # ================ semenatic search ================
        faiss_vectorstore = FAISS.from_texts(self.data_class.recipes_df['ingredients'],
                                             self.embedding_class.embedding)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
        document_lst = faiss_retriever.get_relevant_documents(ingredients_string)
        recipe_string_semantic = ""
        for idx, document in enumerate(document_lst, 1):
            extracted_title = self.get_title(document)
            recipe_string_semantic += f"{idx}. {extracted_title}\n"
            if idx >= display_titles:
                break
        recipe_string_semantic = recipe_string_semantic.strip('\n')
        print("--- titles (semantic) ---")
        print(recipe_string_semantic)

        # ================ ensemble search ================
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                               weights=[0.5, 0.5])
        document_lst = ensemble_retriever.get_relevant_documents(ingredients_string)
        recipe_string_ensemble = ""
        for idx, document in enumerate(document_lst, 1):
            extracted_title = self.get_title(document)
            recipe_string_ensemble += f"{idx}. {extracted_title}\n"
            if idx >= display_titles:
                break
        recipe_string_ensemble = recipe_string_ensemble.strip('\n')
        print("--- titles (ensemble) ---")
        print(recipe_string_ensemble)

        # ================ return ================
        return "", recipe_string_ensemble, ingredients_string, f"recipes successfully retrieved"

    # LLM parse ocr to recipe (using mmocr to run locally)
    def generate_ocr_prompt(self, ocr_text):
        print(f"generate_ocr_prompt(self, ocr_text)")
        ocr_to_recipe_prompt = f"""
You are an expert at decoding OCR text for recipe.
Given a raw text detected by OCR, you rewrite the recipe to the best that is human readable.

Example
Input OCR texts:
```
2009 paghet ladd age choepe
creamy ement untaiko parta
ftspr red dpeppors aales
pcs mentako 8
scrape out toe s rad monband
hbsp i o oking ot
cup heavy cream halfmik half foeam
a daha w2
clover amic mihg ned
heat oil onion atic imotein red
med edium 8lby sed
peppe flakes
add ream ment ataiko cup pahay awates
codlc 1 patg resene cup pata wall
or wey
```
Decoded Recipe:
```
Creamy Mentaiko Pasta

200g spaghetti 
add 18g cheese
1 tbsp cooking oil
1/2 medium onion, sliced
2 cloves garlic, minced
1/2 tsp red pepper flakes
1 cup heavy cream (half milk, half cream)
2 pcs mentaiko (80 g)

1) Scrape out row + discard membrane
2) cook pasta + reserve 1 cup pasta water
3) heat oil, onion, garlic, protein, red
pepper flakes.
4) Add cream, Mentaiko, 1/2 cup pasta water
stir well
5) add pasta
```
"""
        ocr_to_recipe_prompt += f"""
Decode the following OCR to an actual recipe:
```
{ocr_text}
```"""
        print(ocr_to_recipe_prompt)
        return ocr_to_recipe_prompt

    def generate_ocr_prompt_openai(self):
        print(f"generate_ocr_prompt(self, ocr_text)")
        system_prompt = f"""
You are an expert at decoding OCR text for recipe.
Given a raw text detected by OCR, you rewrite the recipe to the best that is human readable.
""".strip()

        few_shot_user_v1 = f"""
2009 paghet ladd age choepe
creamy ement untaiko parta
ftspr red dpeppors aales
pcs mentako 8
scrape out toe s rad monband
hbsp i o oking ot
cup heavy cream halfmik half foeam
a daha w2
clover amic mihg ned
heat oil onion atic imotein red
med edium 8lby sed
peppe flakes
add ream ment ataiko cup pahay awates
codlc 1 patg resene cup pata wall
or wey
""".strip()
        few_shot_response_v1 = f"""
Creamy Mentaiko Pasta

200g spaghetti 
add 18g cheese
1 tbsp cooking oil
1/2 medium onion, sliced
2 cloves garlic, minced
1/2 tsp red pepper flakes
1 cup heavy cream (half milk, half cream)
2 pcs mentaiko (80 g)

1) Scrape out row discard membrane
2) cook pasta reserve 1 cup pasta water
3) heat oil, onion, garlic, protein, red
pepper flakes.
4) Add cream, Mentaiko, 1/2 cup pasta water
stir well
5) add pasta
""".strip()
        
        few_shot_user_v2 = f"""
cacb pepe fv
075 cyp pec econo romo aand phg dwth
jmganeorogest a
salt
tcup pam1 11agh he 9 9319hs
com1 nome cheees aed heppey
fhwe pala in dow
makke a thick pehl
wat
refene gmp pista that
thak pestet is skillet the i jone pyfe w av
hoy pepper
petv ayl htep should
mash w enough wold was
fduryddd cheese
aadoly one a a pet
bvw
oe ngo goundy a comy pata
""".strip()
        few_shot_response_v2 = f"""
cacio e pepe (for 2)
salt
0.75 cup pecorino romano, plus dusting
1/2 cup pamiagino reggiano
1 tbsp pepper
linguine / spaghetti
1) In a bowl, combine bowl cheeses & black pepper
mash w enough cold water to make a thick paste
2) reserve cup of pasta water
3) transfer pasta into bowl
stir vigorously coat pasta
adding tsp of olive oil pasta
water
Peter says step 1 should be
transfer pasta to skillet w some pasta water
slowly add cheese
""".strip()


        few_shot_dict = {}
        few_shot_dict[1] = {
            'user': few_shot_user_v1,
            'assistant': few_shot_response_v1,
        }
        few_shot_dict[2] = {
            'user': few_shot_user_v2,
            'assistant': few_shot_response_v2,
        }
        return system_prompt, few_shot_dict

    def ocr_to_recipe_mmocr(self, ocr_text):
        try:
            print(f"ocr_to_recipe(self, ocr_text)")
            upload_status = "Failed to identify recipe."

            # LLM req/resp
            # ocr_to_recipe_prompt = self.generate_ocr_prompt(ocr_text)
            # predicted_recipe = self.chat_bot_class.chat_bot(ocr_to_recipe_prompt)
            system_prompt, few_shot_dict = self.generate_ocr_prompt_openai()
            openai.api_key = os.getenv("OPENAI_API_KEY")
            completions = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": few_shot_dict[1]["user"]},
                    {"role": "assistant", "content": few_shot_dict[1]["assistant"]},
                    {"role": "user", "content": few_shot_dict[2]["user"]},
                    {"role": "assistant", "content": few_shot_dict[2]["assistant"]},
                    {"role": "user", "content": ocr_text},
                ]
            )
            print(completions.choices[0].message)
            predicted_recipe = completions.choices[0].message.content

            # hardcode processing
            predicted_recipe = predicted_recipe.strip(' \n')
            new_recipe_title, new_recipe_steps = predicted_recipe.split('\n', 1)
            new_recipe_steps = new_recipe_steps.strip(' \n')
            upload_status = "Successfully identified recipe."
            print(f"===\npredicted_recipe = \n{predicted_recipe}\n===")
            print(f"===\nnew_recipe_title = {new_recipe_title}\n===")
            print(f"===\nnew_recipe_steps = {new_recipe_steps}\n===")

        except:
            traceback.print_exc()
            pdb.set_trace()
        return new_recipe_title, new_recipe_steps, upload_status

    # LLM parse ocr to recipe (using mmocr to run locally)
    def ocr_to_recipe(self, ocr_text):
        try:
            print(f"ocr_to_recipe()")

            # LLM req/resp
            system_prompt = "You are an expert chef who who always returns your answer in JSON."
            user_prompt  = "From the given text, identify the title of the recipe, required ingredients, and the step to make it."
            user_prompt += "\nReturn in under 'recipe title' , 'ingredients' and 'recipe steps'."
            user_prompt += f"\n```\n{ocr_text}\n```"

            # Construct payload
            model="gpt-3.5-turbo"
            model="gpt-3.5-turbo-1106"
            response_format={"type": "json_object"}
            payload = {
                "model": model,
                "response_format":response_format,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt }
                ],
                "max_tokens": 300
            }

            # Send to openai
            openai_chat_completions_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }

            response = requests.post(openai_chat_completions_url,
                                     headers=headers,
                                     json=payload
                                    )

            response_str = response.json()['choices'][0]['message']['content']
            response_dict = json.loads(response_str)
            print(f"response_dict = {response_dict}")

            new_recipe_title = response_dict['recipe title']
            ingredients = response_dict['ingredients']
            recipe_steps = response_dict['recipe steps']

            # hardcode processing
            ingredients_str = "\n".join(f'- {ingredient}' for ingredient in ingredients)
            recipe_steps_str = "\n".join(f'- {recipe_step}' for recipe_step in recipe_steps)
            new_recipe_steps = f"Ingredients:\n{ingredients_str}\n\nInstructions:\n{recipe_steps_str}"
            upload_status = "Successfully identified recipe."
            print(f"===\nnew_recipe_title = {new_recipe_title}\n===")
            print(f"===\nnew_recipe_steps = {new_recipe_steps}\n===")

        except:
            traceback.print_exc()
            pdb.set_trace()

        return new_recipe_title, new_recipe_steps, upload_status



    # pdb
    def pdb(self, new_recipe_title, new_recipe_steps, ingredients_string):
        upload_status = ""
        pdb.set_trace()
        return new_recipe_title, new_recipe_steps, ingredients_string, upload_status
