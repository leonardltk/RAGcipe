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

class RecipeEmbeddings():
    def __init__(self, 
                 embedding_model_name, 
                 CHUNKS_TXT, CHUNK_SIZE, CHUNK_OVERLAP
                 ) -> None:

        # global params
        self.CHUNKS_TXT = CHUNKS_TXT
        self.CHUNK_SIZE = CHUNK_SIZE
        self.CHUNK_OVERLAP = CHUNK_OVERLAP

        # Model params
        self.embedding_model_name = embedding_model_name
        self.embedding_cost_per_token = 0
        self.model_kwargs = {'device': 'cpu'} # {"device": "cuda"}

        # to be initialised
        self.embedding = None
        self.vectordb = None

    def load_model(self):
        # Load model
        self.embedding = HuggingFaceInstructEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=self.model_kwargs,
            )

    # ---
    def write_to_chunks_txt(self, chunks_df):
        # Write to file <- chunks_df
        with open(self.CHUNKS_TXT, 'w', encoding='utf-8') as f:
            for page_content in chunks_df['page_content']:
                f.write(f"\n{page_content}\n")
            print(f'Finished writing to {self.CHUNKS_TXT}')

    def update_chunks_csv_df(self, chroma_dir):
        chunks_csv = os.path.join(chroma_dir, 'chunks.csv')
        # make chunks dict
        chunks_dict = {'ids':[], 'page_content':[], 'metadata':[]}
        for ids, page_content, metadata in zip(
            self.vectordb.get()['ids'], 
            self.vectordb.get()['documents'], 
            self.vectordb.get()['metadatas']
        ):
            chunks_dict['ids'].append(ids)
            chunks_dict['page_content'].append(page_content)
            chunks_dict['metadata'].append(metadata)
        # make chunks df
        chunks_df = pd.DataFrame(chunks_dict)
        # make chunks csv
        chunks_df.to_csv(chunks_csv, index=False)
        # update chunks.txt
        self.write_to_chunks_txt(chunks_df)


    # ---
    def make_vector_db(self, split_documents, chroma_dir, force):
        # load from disk
        if (not force) and os.path.exists(chroma_dir):
            print(f'make_vector_db: Table already created, skipping...')
            self.vectordb = Chroma(
                persist_directory=chroma_dir, 
                embedding_function=self.embedding
                )
            print(f'make_vector_db: self.vectordb initialised')
            return
    
        # remove old database files if any
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
    
        # Init
        os.makedirs(chroma_dir)
        chunks_csv = os.path.join(chroma_dir, 'chunks.csv')
    
        # store into csv
        if len(split_documents):
            chunks_df = pd.DataFrame({
                'page_content': [i.page_content for i in split_documents],
                'metadata': [i.metadata for i in split_documents],
            })
        else:
            chunks_df = pd.DataFrame({
                'page_content': [],
                'metadata': [],
            })
        chunks_df.to_csv(chunks_csv, index=False)
    
        # Write to file
        self.write_to_chunks_txt(chunks_df)
    
        # store the database
        print(f'split_documents {split_documents}')
        print(f'make_vector_db: storing split documents into {chroma_dir}')
        vectordb = Chroma.from_texts(
            texts=split_documents,
            embedding=self.embedding,
            persist_directory=chroma_dir
        )
        number_of_database_points = vectordb._collection.count()
        print(f'make_vector_db: number_of_database_points = {number_of_database_points}')
    
        return vectordb

    # ---
    def split_document(self, document_path, chunk_size, chunk_overlap,):
            
        # Initialise the loader
        if '.pdf' in document_path:
            loader = PyPDFLoader(document_path)
        else:
            loader = TextLoader(document_path)

        # Loading the document
        docs = loader.load()

        # Initialise the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )

        # Perform the split
        document_lst = text_splitter.split_documents(docs) # list of Documents()

        return document_lst

    def document_to_vectordb(self, filename, chroma_dir_input, table_mode):
        print(colored('document_to_vectordb:', 'yellow', attrs=['bold']))

        if table_mode == 'create':
            chroma_dir = chroma_dir_input
        elif table_mode == 'add':
            chroma_dir = chroma_dir_input.rstrip('/') + '_new'

        # Split the document into chunks
        split_documents = self.split_document(filename, 
                                              chunk_size=self.CHUNK_SIZE, 
                                              chunk_overlap=self.CHUNK_OVERLAP,
                                              )

        # Post process
        for idx, chunk in enumerate(split_documents):
            chunk.page_content = chunk.page_content.replace('\n', ' ') 
            chunk.page_content = f"{idx}. {chunk.page_content}"

        # Store to table
        vectordb_to_add = self.make_vector_db(split_documents,
                                              chroma_dir,
                                              force=True)

        # Create or Add to current table
        if table_mode == 'create':
            self.vectordb = vectordb_to_add
        elif table_mode == 'add':
            # check for duplicates
            seen_documents = self.vectordb.get()['documents']
            for idx, documents in enumerate(vectordb_to_add.get()['documents']):
                if documents in seen_documents:
                    continue
                print(f'adding {idx} | {documents}')
                self.vectordb._collection.add(
                    ids=vectordb_to_add.get()['ids'][idx],
                    documents=vectordb_to_add.get()['documents'][idx],
                    metadatas=vectordb_to_add.get()['metadatas'][idx],
                )

            # update chunks.csv
            self.update_chunks_csv_df(chroma_dir_input)

        # status message
        status_msg = f'Completed generating {filename} to {chroma_dir}'

        # So next time we don't need to re-generate, just load
        self.vectordb.persist()

        return status_msg


class RecipeEmbeddingsEasy():
    def __init__(self, 
                 embedding_model_name, 
                 CHUNKS_TXT,
                 CHROMA_DIR,
                 RETRIEVER_KWARGS,
                ) -> None:
        """
        CHROMA_DIR = 'docs/chroma/'
        UPLOAD_FOLDER = 'docs/uploads/'
        CHUNKS_TXT = 'docs/chunks.txt'

        RETRIEVER_KWARGS = {
            'search_type': 'similarity', # {similarity, similarity_score_threshold, mmr}
            'samples': 5
        }
        """

        # global params
        self.CHUNKS_TXT = CHUNKS_TXT
        self.CHROMA_DIR = CHROMA_DIR

        # Model params
        self.embedding_model_name = embedding_model_name
        self.model_kwargs = {'device': 'cpu'} # {"device": "cuda"}
        self.vector_kwargs = RETRIEVER_KWARGS

        # to be initialised
        self.embedding = None
        self.vector_db = None
        
        # init it now
        self.load_model()

    def load_model(self):
        # Load model
        self.embedding = HuggingFaceInstructEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=self.model_kwargs,
        )
        # self.embedding = OpenAIEmbeddings()

    # --- vectordb
    def create_vector_db(self, split_documents):
        # remove old database files if any
        if os.path.exists(self.CHROMA_DIR):
            shutil.rmtree(self.CHROMA_DIR)
    
        # make the directory
        os.makedirs(self.CHROMA_DIR)

        # # split_documents
        self.vector_db = Chroma.from_documents(
            documents=split_documents,
            embedding=self.embedding,
            persist_directory=self.CHROMA_DIR
        )

        self.vector_db.persist()

        number_of_database_points = self.vector_db._collection.count()
        print(f'self.vector_db: number_of_database_points = {number_of_database_points}')

    def read_vector_db(self):
        self.vector_db = Chroma(
            persist_directory=self.CHROMA_DIR, 
            embedding_function=self.embedding
        )

        number_of_database_points = self.vector_db._collection.count()
        print(f'self.vector_db: number_of_database_points = {number_of_database_points}')

    def add_vector_db(self, split_documents):
        print("\tadd_vector_db()")
        print(f'\t\tBefore: number_of_database_points = {self.vector_db._collection.count()}')

        # Make a vector db to add
        vectordb_to_add = Chroma.from_documents(
            documents=split_documents,
            embedding=self.embedding,
            persist_directory=None,
        )
        print(f"\t\tSucessfully generated vectordb_to_add")

        # Add to vector_db
        seen_documents = self.vector_db.get()['documents']
        for idx, documents in enumerate(vectordb_to_add.get()['documents']):
            print(f'\t\t\tadding {idx} | {documents}')
            self.vector_db._collection.add(
                ids=vectordb_to_add.get()['ids'][idx],
                documents=vectordb_to_add.get()['documents'][idx],
                metadatas=vectordb_to_add.get()['metadatas'][idx],
            )
        print(f"\t\tSucessfully added to self.vector_db")

        print(f'\t\tAfter: number_of_database_points = {self.vector_db._collection.count()}')

    def remove_vector_db(self, recipe_title):
        print("\tremove_vector_db()")
        print(f'\t\tBefore: number_of_database_points = {self.vector_db._collection.count()}')

        for idx, documents in enumerate(self.vector_db.get()['documents']):
            if documents == recipe_title:
                delete_id = self.vector_db.get()['ids'][idx]
                delete_metadatas = self.vector_db.get()['metadatas'][idx]
                print(f"\t\t\tremoving {recipe_title} self.vector_db ...")
                assert delete_metadatas['title'] == recipe_title
                # print('delete_id', delete_id)
                # print('delete_metadatas', delete_metadatas)
                self.vector_db._collection.delete([delete_id])
                print(f"\t\t\tdone.")
        print(f"\t\tSucessfully removed from self.vector_db")

        print(f'\t\tBefore: number_of_database_points = {self.vector_db._collection.count()}')

    # ----- context/citation -----
    def get_documents(self, standalone_question):
        search_type = self.vector_kwargs['search_type']
        samples = self.vector_kwargs['samples']

        # get documents
        if search_type == 'similarity':
            documents = self.vector_db.similarity_search(standalone_question, k=samples )
        elif search_type == 'similarity_score_threshold':
            score_threshold = self.vector_kwargs['score']
            context_with_score = self.vector_db.similarity_search_with_score(standalone_question, k=samples )
            documents = [i[0] for i in context_with_score if i[1]>=score_threshold]
        elif search_type == 'mmr':
            documents = self.vector_db.max_marginal_relevance_search(standalone_question, k=samples, fetch_k=samples*2)
        else:
            documents = []
        
        return documents

    def _get_context(self, standalone_question):
        documents = self.get_documents(standalone_question)
        # get context as string
        context_lst = [d.page_content for d in documents]
        context_string = '\n\n'.join(context_lst)

        if 0:# get citations
            citation_dict = {}
            for i in documents:
                filename = i.metadata['source']
                page_number = i.metadata['page']
                # update locally
                if not filename in citation_dict:
                    citation_dict[filename] = set()
                citation_dict[filename].add(page_number)
                # update globally
                if not filename in self.total_citation_dict:
                    self.total_citation_dict[filename] = set()
                self.total_citation_dict[filename].add(page_number)
            # collate to list
            citations_lst = []
            for filename, page_numbers in citation_dict.items():
                citation = f"{filename} | pages={page_numbers}"
                citations_lst.append(citation)
            # condense to string
            citations = '\n'.join(citations_lst)

        return documents, context_string # , citations

