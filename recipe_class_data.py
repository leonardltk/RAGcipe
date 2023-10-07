import traceback
import pdb
import pprint

from fuzzywuzzy import fuzz
import Levenshtein
import pandas as pd

from langchain.schema import Document
from langchain.chains.query_constructor.base import AttributeInfo


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

        titles_documents = self.df_to_documents(add_recipes_df,
                                                page_content_column="title",
                                                metadata_lst=["title", "cuisine", "Carbs", "Proteins"])

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

        return titles_documents

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

    # sanity checks
    def sanity_check(self, mode, kwargs_dict):
        """
        kwargs_dict={
            'recipe_title':recipe_title,
            'recipe_steps':recipe_steps
        }
        """
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


