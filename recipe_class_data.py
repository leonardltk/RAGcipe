import traceback
import pdb
import pprint
import re

from fuzzywuzzy import fuzz
import Levenshtein
import pandas as pd

from langchain.schema import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain import PromptTemplate


class RecipeData():
    def __init__(self, recipes_csv):
        self.recipes_csv = recipes_csv

        # self.allowed_cuisine_lst = [ "Seafood", "Vegetarian", "Vegan", "Western", "European", "North American", "Italian", "French", "Spanish", "Greek", "Mediterranean", "Japanese", "Chinese", "Korean", "Thai", "Vietnamese", "Indian", "Mexican", "Middle Eastern", "African", "South American", "Caribbean", "Barbecue/Grilled", "Fast Food", "Fusion", ]
        self.allowed_cuisine_lst = ["none", "japanese", "italian"]
        self.allowed_carbohydrates_lst = ["none", "noodles", "rice", "pasta"]
        self.allowed_proteins_lst = ["none", "chicken", "seafood"]
        # set up the prompt config
        self.categories_extraction_prompt_template = self.setup_prompt_categories_extraction()
        self.ingredients_extraction_prompt_template = self.setup_prompt_ingredients_extraction()
        self.metadata_field_info, self.document_content_description = self.self_querying_retriever()

        self.recipes_df = pd.read_csv(self.recipes_csv)

        self.recipes_dict = dict(
            (row['title'], row['recipe'])
            for idx, row in self.recipes_df.iterrows()
        )

    # dataprep
    def df_to_documents(self, recipes_df, page_content_column, metadata_lst):
        
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

        titles_documents = self.df_to_documents(self.recipes_df,
                                                page_content_column="title",
                                                metadata_lst=["title", "cuisine", "Carbs", "Proteins"])
        ingredients_documents = self.df_to_documents(self.recipes_df,
                                                     page_content_column="ingredients",
                                                     metadata_lst=["title"])

        return titles_documents, ingredients_documents

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

        # Check existence
        if recipe_title in self.recipes_dict:
            status_message = f"{recipe_title} already exists."
            pdb.set_trace()
            _ = 'TO DO: deal with this next'

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

        # extract documents
        titles_documents = self.df_to_documents(add_recipes_df,
                                                page_content_column="title",
                                                metadata_lst=["title", "cuisine", "Carbs", "Proteins"])
        ingredients_documents = self.df_to_documents(add_recipes_df,
                                                     page_content_column="ingredients",
                                                     metadata_lst=["title"])

        return titles_documents, ingredients_documents

    def modify_data(self, modify_recipes_df):
        print("\tmodify_data(recipe_title, recipe_steps)")
        status_fail_bool = None
        status_message = None

        recipe_title = modify_recipes_df['title'][0]
        recipe_steps = modify_recipes_df['recipe'][0]

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
                for column_name in list(self.recipes_df):
                    self.recipes_df[column_name][idx] = modify_recipes_df[column_name][0]
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

        # extract documents
        titles_documents = self.df_to_documents(modify_recipes_df,
                                                page_content_column="title",
                                                metadata_lst=["title", "cuisine", "Carbs", "Proteins"])
        ingredients_documents = self.df_to_documents(modify_recipes_df,
                                                     page_content_column="ingredients",
                                                     metadata_lst=["title"])

        return titles_documents, ingredients_documents

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

    def sanity_check(self, mode, kwargs_dict):
        '''
        kwargs_dict={
            'recipe_title':recipe_title,
            'recipe_steps':recipe_steps
        }
        '''
        if mode == 'add':
            # recipes_csv, recipes_df
            for idx, tmp_df in enumerate([
                    pd.read_csv(self.recipes_csv),
                    self.recipes_df.copy()
                ]):
                print(f'idx={idx}')
                # tmp_df
                index_list = list(tmp_df.index)
                assert len(index_list) == len(set(index_list))
                # this_recipe_df
                this_recipe_df =  tmp_df[tmp_df['title']==kwargs_dict['recipe_title']]
                assert len(this_recipe_df) == 1
                for idx, row in this_recipe_df.iterrows():
                    assert row['recipe'] == kwargs_dict['recipe_steps']
            # recipes_dict
            assert kwargs_dict['recipe_steps'] == self.recipes_dict.get(kwargs_dict['recipe_title'], '')
        elif mode == 'remove':
            # recipes_csv, recipes_df
            for idx, tmp_df in enumerate([
                    pd.read_csv(self.recipes_csv),
                    self.recipes_df.copy()
                ]):
                print(f'idx={idx}')
                # tmp_df
                index_list = list(tmp_df.index)
                assert len(index_list) == len(set(index_list))
                # this_recipe_df
                this_recipe_df =  tmp_df[tmp_df['title']==kwargs_dict['recipe_title']]
                assert len(this_recipe_df) == 0
            # recipes_dict
            assert '' == self.recipes_dict.get(kwargs_dict['recipe_title'], '')
        elif mode == 'modify':
            # recipes_csv, recipes_df
            for idx, tmp_df in enumerate([
                    pd.read_csv(self.recipes_csv),
                    self.recipes_df.copy(),
                ]):
                print(f'idx={idx}')
                # tmp_df
                index_list = list(tmp_df.index)
                assert len(index_list) == len(set(index_list))
                # this_recipe_df
                this_recipe_df =  tmp_df[tmp_df['title']==kwargs_dict['recipe_title']]
                assert len(this_recipe_df) == 1
                for idx, row in this_recipe_df.iterrows():
                    assert row['recipe'] == kwargs_dict['recipe_steps']
            # recipes_dict
            assert kwargs_dict['recipe_steps'] == self.recipes_dict.get(kwargs_dict['recipe_title'], '')

    # formulate prompts
    def setup_prompt_categories_extraction(self, ):
        # initialise few shot examples
        recipe_example_1 = f'''
        Chicken Udon Noodle Soup
            1. cook udon noodles according to directions (al dente)
            2. Place
                - 1 tsp dashi
                - 1 tsp soy sauce
                - 1 pinch salt
                - 1 pinch sugar
                - 1 cupo boiling wateer into a deep bowl
            3. slide cooked udon into soup
            4. add cooked chicken
        '''
        recipe_example_1 = re.sub(r'[ ]+', ' ', recipe_example_1).strip('\n')
        response_example_1 = '''{{{{
            "Cuisines": "japanese",
            "Carbohydrates": "noodles",
            "Proteins": "chicken"
        }}}}'''
        response_example_1 = re.sub(r'[ ]+', ' ', response_example_1).strip('\n')

        recipe_example_2 = '''
        Prawn Pesto Pasta
            1. Take 12-14 prawns. Thaw and dry them
            2. Marinate prawns with Salt, pepper, herbs, chilli powder
            3. 1 clove garlic(minced) & olive oil the pan

            4. Throw prawns in. Add chilli powder again
            5. Some butter (1-2 spoonful)
            6. Throw pasta in

            7. For each plate, Add 1 spoonful of pesto paste, and some pasta water.
            8. Put the cooked prawns and spaghetti in.
        '''
        recipe_example_2 = re.sub(r'[ ]+', ' ', recipe_example_2).strip('\n')
        response_example_2 = '''{{{{
            "Cuisines": "italian",
            "Carbohydrates": "pasta",
            "Proteins": "seafood"
        }}}}'''
        response_example_2 = re.sub(r'[ ]+', ' ', response_example_2).strip('\n')

        # set up system prompt
        template = f'''
            You are an expert chef who reads the recipe,
            and extracts desired key information for each categories.
                - Cuisines: choose only one from {self.allowed_cuisine_lst}
                - Carbohydrates: choose only one from {self.allowed_carbohydrates_lst}
                - Proteins: choose only one from {self.allowed_proteins_lst}
            Return them in the following specified json form.

            Here are some examples:
            << Example 1. >>
            Recipe:
            ```
            {recipe_example_1}
            ```
            AI: {response_example_1}
            << Example 2. >>
            Recipe:
            ```
            {recipe_example_2}
            ```
            AI: {response_example_2}

            Now extract the information in json form:
        '''
        template = template + '''
            Recipe:
            ```
            {recipe_title}
            {recipe_steps}
            ```
            AI:
            '''
        template = re.sub(r'[ ]+', ' ', template).strip('\n')

        # format prompt template
        prompt_template = PromptTemplate(template=template,
                                         input_variables=["recipe_title", "recipe_steps"])
        return prompt_template

    def setup_prompt_ingredients_extraction(self, ):
        # initialise few shot examples
        recipe_example_1 = f'''
        Chicken Udon Noodle Soup
            1. cook udon noodles according to directions (al dente)
            2. Place
                - 1 tsp dashi
                - 1 tsp soy sauce
                - 1 pinch salt
                - 1 pinch sugar
                - 1 cupo boiling wateer into a deep bowl
            3. slide cooked udon into soup
            4. add cooked chicken
        '''
        recipe_example_1 = re.sub(r'[ ]+', ' ', recipe_example_1).strip('\n')
        response_example_1 = '''
            Udon noodles
            dashi
            soy sauce
            salt
            sugar
            chicken
        '''
        response_example_1 = re.sub(r'[ ]+', ' ', response_example_1).strip('\n')
        response_example_1 = response_example_1.lower()

        recipe_example_2 = '''
        Prawn Pesto Pasta
            1. Take 12-14 prawns. Thaw and dry them
            2. Marinate prawns with Salt, pepper, herbs, chilli powder
            3. 1 clove garlic(minced) & olive oil the pan

            4. Throw prawns in. Add chilli powder again
            5. Some butter (1-2 spoonful)
            6. Throw pasta in

            7. For each plate, Add 1 spoonful of pesto paste, and some pasta water.
            8. Put the cooked prawns and spaghetti in.
        '''
        recipe_example_2 = re.sub(r'[ ]+', ' ', recipe_example_2).strip('\n')
        response_example_2 = '''
            prawns
            Salt
            Pepper
            Herbs
            Chilli powder
            garlic
            Olive oil
            Butter
            Pasta
            Pesto
        '''
        response_example_2 = re.sub(r'[ ]+', ' ', response_example_2).strip('\n')
        response_example_2 = response_example_2.lower()

        # set up system prompt
        template = f'''
            You are an expert chef who reads the recipe,
            and extracts essential ingredients.

            Here are some examples:
            << Example 1. >>
            Recipe:
            ```
            {recipe_example_1}
            ```
            AI: {response_example_1}
            << Example 2. >>
            Recipe:
            ```
            {recipe_example_2}
            ```
            AI: {response_example_2}

            Now extract the ingredients, delimited by newline:
        '''
        template = template + '''
            Recipe:
            ```
            {recipe_title}
            {recipe_steps}
            ```
            AI:
            '''
        template = re.sub(r'[ ]+', ' ', template).strip('\n')

        # format prompt template
        prompt_template = PromptTemplate(template=template,
                                         input_variables=["recipe_title", "recipe_steps"])
        return prompt_template
