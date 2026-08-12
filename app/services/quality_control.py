import logging
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.llm_provider import LLMProviderFactory

logger = logging.getLogger(__name__)

_EMBEDDER_CACHE = None

class QualityController:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        global _EMBEDDER_CACHE
        if _EMBEDDER_CACHE is None:
            try:
                _EMBEDDER_CACHE = SentenceTransformer(embedding_model)
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                _EMBEDDER_CACHE = None
        
        self.embedder = _EMBEDDER_CACHE
        self.llm_factory = None # Lazy init to avoid circular deps or unnecessary init

    def deduplicate(self, data: List[Dict[str, Any]], keys: List[str] = None, threshold: float = 0.90) -> List[Dict[str, Any]]:
        """
        Removes duplicate entries based on semantic similarity of specified keys.
        If keys is None, uses all string values.
        """
        if not data or not self.embedder:
            return data
        
        try:
            # Extract text to embed
            texts = []
            for item in data:
                if keys:
                    text_parts = [str(item.get(k, "")) for k in keys]
                    texts.append(" ".join(text_parts))
                else:
                    # Use all values
                    texts.append(" ".join([str(v) for v in item.values()]))
            
            if not texts:
                return data

            embeddings = self.embedder.encode(texts)
            
            # Calculate cosine similarity matrix
            sim_matrix = cosine_similarity(embeddings)
            
            # Mask upper triangle
            # We want to keep the first occurrence and drop subsequent ones that use it
            keep_indices = []
            dropped_indices = set()
            
            for i in range(len(data)):
                if i in dropped_indices:
                    continue
                
                keep_indices.append(i)
                
                # Check subsequent items
                for j in range(i + 1, len(data)):
                    if j in dropped_indices:
                        continue
                    
                    if sim_matrix[i][j] >= threshold:
                        dropped_indices.add(j)
                        # logger.debug(f"Dropping item {j} (similar to {i}, score {sim_matrix[i][j]:.2f})")
            
            logger.info(f"Deduplication: Kept {len(keep_indices)} out of {len(data)}")
            return [data[i] for i in keep_indices]
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return data

    async def check_hallucination(self, item: Dict[str, Any], context: str, provider: str = "groq") -> bool:
        """
        Uses LLM to verify if the item content (e.g., answer) is supported by the context.
        Returns True if consistent (not hallucinated), False otherwise.
        """
        # This is expensive, so use sparingly or with cheaper models.
        if not context:
            return True
            
        try:
            if not self.llm_factory:
                self.llm_factory = LLMProviderFactory(provider)
            
            llm = self.llm_factory.llm
            
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            prompt = ChatPromptTemplate.from_template(
                """
                You are a fact-checking assistant.
                Context: {context}
                
                Statement/Q&A to check:
                {item}
                
                Task: Determine if the information in the Statement is supported by the Context.
                Return only "supported" or "unsupported".
                """
            )
            
            chain = prompt | llm | StrOutputParser()
            
            result = await chain.ainvoke({"context": context[:2000], "item": str(item)})
            
            is_supported = "supported" in result.lower() and "unsupported" not in result.lower()
            return is_supported
            
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return True # Fail open

