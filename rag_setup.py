import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RAGManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        # Using a small, fast open-source embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

    def initialize_db(self):
        """Initializes the database with some sample educational content."""
        sample_documents = [
            Document(
                page_content="Linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables.",
                metadata={"topic": "Machine Learning", "source": "Intro to ML"}
            ),
            Document(
                page_content="In Python, a dictionary is a collection which is unordered, changeable and indexed. In Python dictionaries are written with curly brackets, and they have keys and values.",
                metadata={"topic": "Python Programming", "source": "Python Basics"}
            ),
            Document(
                page_content="The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides.",
                metadata={"topic": "Mathematics", "source": "Geometry 101"}
            ),
            Document(
                page_content="Time management for studying: The Pomodoro Technique is a time management method that uses a timer to break down work into intervals, traditionally 25 minutes in length, separated by short breaks.",
                metadata={"topic": "Study Skills", "source": "Effective Learning"}
            ),
            Document(
                page_content="K-Means clustering is an unsupervised machine learning algorithm that groups data into k number of clusters based on similarity.",
                metadata={"topic": "Machine Learning", "source": "Intro to ML"}
            )
        ]
        
        # Initialize or load Chroma
        if not os.path.exists(self.persist_directory):
            print("Creating new vector store...")
            self.vector_store = Chroma.from_documents(
                documents=sample_documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
            
    def get_retriever(self, k=2):
        if not self.vector_store:
            self.initialize_db()
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def search(self, query, k=2):
        if not self.vector_store:
            self.initialize_db()
        results = self.vector_store.similarity_search(query, k=k)
        return [{"content": r.page_content, "source": r.metadata.get("source")} for r in results]
