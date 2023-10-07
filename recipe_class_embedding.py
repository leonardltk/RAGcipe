import os
import shutil
import pprint
import pdb
import traceback
import pandas as pd
from termcolor import colored
import functools
import torch

# import openai
# import tiktoken

import langchain
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline


class RecipeEmbeddingsEasy():
    def __init__(self, 
                 embedding_model_name, 
                 CHROMA_DIR,
                 RETRIEVER_KWARGS,
                ) -> None:
        """
        CHROMA_DIR = 'docs/chroma/'
        UPLOAD_FOLDER = 'docs/uploads/'

        RETRIEVER_KWARGS = {
            'search_type': 'similarity', # {similarity, similarity_score_threshold, mmr}
            'samples': 5
        }
        """

        # global params
        self.CHROMA_DIR = CHROMA_DIR
        self.chroma_titles_dir = os.path.join(self.CHROMA_DIR, 'titles')
        self.chroma_ingredients_dir = os.path.join(self.CHROMA_DIR, 'ingredients')

        # Model params
        self.embedding_model_name = embedding_model_name
        self.model_kwargs = {'device': 'cpu'} # {"device": "cuda"}
        self.vector_kwargs = RETRIEVER_KWARGS

        # embedding model
        self.embedding = self.load_model()

        # vector dbss
        self.vector_dbs = {
            "titles_db": None,
            "ingredients_db": None,
        }

    def load_model(self):
        # Load model

        embedding = HuggingFaceInstructEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=self.model_kwargs,
        )

        # embedding = OpenAIEmbeddings()
        return embedding

    # --- create vectordb
    def document_to_vectordb(self, split_documents, persist_directory):
        vectordb = Chroma.from_documents(
            documents=split_documents,
            embedding=self.embedding,
            persist_directory=persist_directory
        )

        vectordb.persist()

        number_of_database_points = vectordb._collection.count()
        print(f'vectordb: number_of_database_points = {number_of_database_points}')

        return vectordb

    def create_vector_db(self, titles_documents, ingredients_documents):
        # remove old database files if any
        if os.path.exists(self.CHROMA_DIR):
            shutil.rmtree(self.CHROMA_DIR)
    
        # make the directory
        os.makedirs(self.CHROMA_DIR)
        os.makedirs(self.chroma_titles_dir)
        os.makedirs(self.chroma_ingredients_dir)
    
        # make vector dbs
        self.vector_dbs["titles_db"] = self.document_to_vectordb(titles_documents, self.chroma_titles_dir)
        self.vector_dbs["ingredients_db"] = self.document_to_vectordb(ingredients_documents, self.chroma_ingredients_dir)

    # --- read vectordb
    def read_vector_db(self, vector_db_name):
        self.vector_dbs[vector_db_name] = Chroma(
            persist_directory=self.CHROMA_DIR, 
            embedding_function=self.embedding
        )

        number_of_database_points = self.vector_dbs[vector_db_name]._collection.count()
        print(f'self.vector_dbs[vector_db_name]: number_of_database_points = {number_of_database_points}')

    # --- add/remove items from vector db
    def add_vector_db(self, split_documents, vector_db_name):
        print(f"\tadd_vector_db(vector_db_name={vector_db_name})")
        print(f'\t\tBefore: number_of_database_points = {self.vector_dbs[vector_db_name]._collection.count()}')

        # Make a vector db to add
        vectordb_to_add = Chroma.from_documents(
            documents=split_documents,
            embedding=self.embedding,
            persist_directory=None,
        )
        print(f"\t\tSucessfully generated vectordb_to_add")

        # Add to titles_db
        seen_documents = self.vector_dbs[vector_db_name].get()['documents']
        for idx, documents in enumerate(vectordb_to_add.get()['documents']):
            print(f'\t\t\tadding {idx} | {documents}')
            self.vector_dbs[vector_db_name]._collection.add(
                ids=vectordb_to_add.get()['ids'][idx],
                documents=vectordb_to_add.get()['documents'][idx],
                metadatas=vectordb_to_add.get()['metadatas'][idx],
            )
        print(f"\t\tSucessfully added to self.vector_dbs[{vector_db_name}]")

        print(f'\t\tAfter: number_of_database_points = {self.vector_dbs[vector_db_name]._collection.count()}')

    def remove_vector_db(self, recipe_title, vector_db_name):
        print(f"\tremove_vector_db(vector_db_name={vector_db_name})")
        print(f'\t\tBefore: number_of_database_points = {self.vector_dbs[vector_db_name]._collection.count()}')

        for idx, documents in enumerate(self.vector_dbs[vector_db_name].get()['documents']):
            if documents == recipe_title:
                delete_id = self.vector_dbs[vector_db_name].get()['ids'][idx]
                delete_metadatas = self.vector_dbs[vector_db_name].get()['metadatas'][idx]
                print(f"\t\t\tremoving {recipe_title} self.vector_dbs[{vector_db_name}] ...")
                assert delete_metadatas['title'] == recipe_title
                # print('delete_id', delete_id)
                # print('delete_metadatas', delete_metadatas)
                self.vector_dbs[vector_db_name]._collection.delete([delete_id])
                print(f"\t\t\tdone.")
        print(f"\t\tSucessfully removed from self.vector_dbs[{vector_db_name}]")

        print(f'\t\tBefore: number_of_database_points = {self.vector_dbs[vector_db_name]._collection.count()}')

    # sanity checks
    def sanity_check(self, mode, kwargs_dict):
        if mode == 'add':
            # Add to titles_db
            tmp_dict = self.vector_dbs['titles_db'].get()
            # documents
            document_lst = tmp_dict['documents']
            assert len(document_lst) == len(set(document_lst))
            assert kwargs_dict['recipe_title'] in document_lst
            # metadatas
        elif mode == 'remove':
            # Add to titles_db
            tmp_dict = self.vector_dbs["titles_db"].get()
            # documents
            document_lst = tmp_dict['documents']
            assert len(document_lst) == len(set(document_lst))
            assert not kwargs_dict['recipe_title'] in document_lst
            # metadatas
        elif mode == 'modify':
            # Add to titles_db
            tmp_dict = self.vector_dbs['titles_db'].get()
            # documents
            document_lst = tmp_dict['documents']
            assert len(document_lst) == len(set(document_lst))
            assert kwargs_dict['recipe_title'] in document_lst
            # metadatas

    # ----- retrieval -----
    def get_documents_old(self, standalone_question):
        search_type = self.vector_kwargs['search_type']
        samples = self.vector_kwargs['samples']

        # get documents
        if search_type == 'similarity':
            documents = self.titles_db.similarity_search(standalone_question, k=samples )
        elif search_type == 'similarity_score_threshold':
            score_threshold = self.vector_kwargs['score']
            context_with_score = self.titles_db.similarity_search_with_score(standalone_question, k=samples )
            documents = [i[0] for i in context_with_score if i[1]>=score_threshold]
        elif search_type == 'mmr':
            documents = self.titles_db.max_marginal_relevance_search(standalone_question, k=samples, fetch_k=samples*2)
        else:
            documents = []
        
        return documents

    def get_documents(self, standalone_question, vector_db_name):
        search_type = self.vector_kwargs['search_type']
        samples = self.vector_kwargs['samples']

        # get documents
        if search_type == 'similarity':
            documents = self.vector_dbs[vector_db_name].similarity_search(standalone_question, k=samples )
        elif search_type == 'similarity_score_threshold':
            score_threshold = self.vector_kwargs['score']
            context_with_score = self.vector_dbs[vector_db_name].similarity_search_with_score(standalone_question, k=samples )
            documents = [i[0] for i in context_with_score if i[1]>=score_threshold]
        elif search_type == 'mmr':
            documents = self.vector_dbs[vector_db_name].max_marginal_relevance_search(standalone_question, k=samples, fetch_k=samples*2)
        else:
            documents = []
        
        return documents

    # def _get_context(self, standalone_question):
    #     documents = self.get_documents(standalone_question)
    #     # get context as string
    #     context_lst = [d.page_content for d in documents]
    #     context_string = '\n\n'.join(context_lst)

    #     if 0:# get citations
    #         citation_dict = {}
    #         for i in documents:
    #             filename = i.metadata['source']
    #             page_number = i.metadata['page']
    #             # update locally
    #             if not filename in citation_dict:
    #                 citation_dict[filename] = set()
    #             citation_dict[filename].add(page_number)
    #             # update globally
    #             if not filename in self.total_citation_dict:
    #                 self.total_citation_dict[filename] = set()
    #             self.total_citation_dict[filename].add(page_number)
    #         # collate to list
    #         citations_lst = []
    #         for filename, page_numbers in citation_dict.items():
    #             citation = f"{filename} | pages={page_numbers}"
    #             citations_lst.append(citation)
    #         # condense to string
    #         citations = '\n'.join(citations_lst)

    #     return documents, context_string # , citations

