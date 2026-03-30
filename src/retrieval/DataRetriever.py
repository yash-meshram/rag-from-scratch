from src.indexing.VectorStore import VectorStore
from src.indexing.Embedding import EmbeddingManager
from typing import List, Dict, Any

class DataRetriever():
    '''Handles query based retriever from the vector store'''
    def __init__(
            self,
            collection_name: str, 
            persist_directory: str,
            model_name: str = "multi-qa-MiniLM-L6-cos-v1"
        ):
        '''
        Initializing the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Embedding manager, will help in generating embedding
        '''
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        self.vector_store = VectorStore(collection_name=self.collection_name, persist_directory=self.persist_directory)
        self.embedding_manager = EmbeddingManager(model_name=self.model_name)
        
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.25) -> List[Dict[str, Any]]:
        '''
        Retrieve relevant documents for a query
        
        Args:
            query: the search query
            top_k: number of top results
            score_threshold: minimum similarity score required
        
        Return: list of dictonaries containing releveant documents and metadata
        '''
        # generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(texts = [query])[0]
        
        # search the query embedding in vectr db
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = top_k
            )
            
            retrieved_docs = []
            rank = 0
            
            if results['documents'] and results['documents'][0]:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for i, (id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': rank + 1
                        })
                    rank += 1
                print(f"Retrieved {len(retrieved_docs)} documents")
            else:
                print("No documents found")
                
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retriever: {e}")
            return []
