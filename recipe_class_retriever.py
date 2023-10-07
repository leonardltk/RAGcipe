import pprint
import pdb
import traceback

import pandas as pd

from langchain.retrievers.self_query.base import SelfQueryRetriever


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

    # def lookup_recipe(self, user_input, past_history):
    #     output_lst = self.retriever.get_relevant_documents(user_input)
    #     response_lst = []
    #     for idx, output in enumerate(output_lst):
    #         response_lst.append(f"{idx}. {output.page_content}")
    #     LLM_response = "\n".join(response_lst)

    #     past_history.append((user_input, LLM_response))

    #     return "", past_history

    # embedding lookup
    def recipe_lookup(self, standalone_question):
        recipe_response_title = "No recipe found"
        recipe_response_steps = ""
        status_message = ""
        SIMILAR_RECIPE_MSG = "\nSimilar recipes:"

        recipes_dict = self.data_class.recipes_dict
        documents = self.embedding_class.get_documents(standalone_question)

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
                else:
                    recipe_response_steps = ""
                    status_message = f"'{recipe_response_title}' not in recipes_dict, please check."
            else:
                recipe_response_title_alternatives = document.metadata['title']
                status_message += f"\n\t{idx}. {recipe_response_title_alternatives}"
        if status_message.endswith(SIMILAR_RECIPE_MSG):
            status_message.replace(SIMILAR_RECIPE_MSG, "")


        return recipe_response_title, recipe_response_steps, status_message

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

        return "", f"{recipe_title} successfully added"

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

        return "", f"{recipe_title} successfully removed"

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

        return "", f"{recipe_title} successfully modified"

    def list_recipe(self):
        print(f'list_recipe()')

        all_titles = sorted(self.data_class.recipes_dict)
        all_titles = [ f'{idx}. {title}' for idx, title in enumerate(all_titles, 1)]
        all_titles = "\n".join(all_titles)

        upload_status = "Success: Listed all recipes from database"

        return all_titles, upload_status
