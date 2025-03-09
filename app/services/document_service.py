
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.logger import logger
from app.exceptions.custom_exceptions import DocumentProcessingException
import io
import requests
import json
from langchain.vectorstores import Qdrant
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import FastEmbedSparse
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
import io
from PyPDF2 import PdfReader
from langchain_qdrant import QdrantVectorStore,RetrievalMode
import io
import uuid
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

class DocumentService:
    def __init__(self):
        self.model_name = "BAAI/bge-large-en"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
        self.sparse_embedding = FastEmbedSparse(
            model_name="Qdrant/bm25",
            batch_size=4,
            cache_dir="cache"
        )
        self.url = "http://localhost:6333"
        self.client = QdrantClient(url=self.url)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        self.collection_info = []

    def process_document(self, file_path):
        """
        Uploads and processes multiple PDF documents.

        Args:
            files (list[UploadFile]): A list of PDF files to upload.

        Returns:
            JSONResponse: A success message if the documents are processed successfully.

        Raises:
            HTTPException: If the file type is invalid or an error occurs during processing.
        """


        try:
            with open(file_path, "rb") as f:
                pdf_file = io.BytesIO(f.read())

            reader = PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''

            docs = self.text_splitter.split_text(text)
            collection_name = str(uuid.uuid4())

            collection = QdrantVectorStore.from_texts(
                texts=docs,
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embedding,
                url=self.url,
                collection_name=collection_name,
                retrieval_mode=RetrievalMode.HYBRID
            )
            self.collection_info.append({
                "name": collection_name,
                "collection": collection
            })
            logger.info(f"collection name: {str(collection_name)}")

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise DocumentProcessingException("Failed to process document.")

    def select_collection(self, query):
        try:
            query_embedding = self.embeddings.embed_query(query)
            query_embedding = self.normalize([query_embedding])[0]

            similarities = {}
            highest_similarity = -1
            best_collection = None
            best_collection_obj = None
            if not self.collection_info:
                client = QdrantClient(host="localhost", port=6333)

                # Retrieve all collections
                collections = client.get_collections()
                collection_info = []
                # Print the collections
                for collection in collections.collections:
                        collection_obj = QdrantVectorStore(
                            embedding=self.embeddings,
                            sparse_embedding=self.sparse_embedding,
                            client=self.client,
                            collection_name=collection.name,
                            retrieval_mode=RetrievalMode.HYBRID  # Use hybrid retrieval mode
                        )
                        self.collection_info.append({
                            "name": collection_obj.collection_name,
                            "collection": collection_obj
                        })
            for collection_info in self.collection_info:
                chunks = collection_info["collection"].similarity_search(query=query, k=3)
                for chunk in chunks:
                    chunk_embedding = self.embeddings.embed_query(chunk.page_content)
                    chunk_embedding = self.normalize([chunk_embedding])[0]
                    similarity_score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]

                    if collection_info["name"] not in similarities:
                        similarities[collection_info["name"]] = []
                    similarities[collection_info["name"]].append(similarity_score)

                    if similarity_score > highest_similarity:
                        highest_similarity = similarity_score
                        best_collection = collection_info["name"]
                        best_collection_obj=collection_info['collection']

            if highest_similarity < 0.8:
                best_collection = False
                highest_similarity = 0.7

            logger.info(f"Highest similarity: {highest_similarity} for collection: {best_collection}")
            return best_collection_obj

        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            raise DocumentProcessingException("Failed to query collection.")

    def parsing_to_llm(self, query, content):
        try:
            full_response = []
            prompt = """
            Extract relevant information from the provided PDF data to answer the following question:
            Question: {user_input}
            Context: {relevant_document}
            Task: Answer the question based on the information provided in the PDF data. If the answer is not explicitly stated, use inference and reasoning to provide a response.
            Response Format: Provide a concise and accurate answer to the question. If necessary, include relevant supporting information or context from the PDF data.
            You can modify this prompt to fit your specific needs and the format of your PDF data.
            """

            url = 'http://localhost:11434/api/generate'
            data = {
                "model": "llama3:8b",
                "prompt": prompt.format(user_input=query, relevant_document=content)
            }
            headers = {'Content-Type': 'application/json'}

            response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

            try:
                for line in response.iter_lines():
                    if line:
                        decoded_line = json.loads(line.decode('utf-8'))
                        full_response.append(decoded_line['response'])
            finally:
                response.close()

            logger.info(f"LLM Response: {''.join(full_response)}")
            return ''.join(full_response)

        except Exception as e:
            logger.error(f"Error parsing to LLM: {str(e)}")
            raise DocumentProcessingException("Failed to parse content to LLM.")

    def normalize(self, embeddings):
        """
        Normalizes the embeddings to unit vectors.

        Args:
            embeddings (list): A list of embeddings to normalize.

        Returns:
            numpy.ndarray: The normalized embeddings.
        """
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)