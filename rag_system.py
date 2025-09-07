"""
RAG (Retrieval-Augmented Generation) system for question answering.
"""

import os
from typing import List, Dict, Any, Optional
import openai
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGSystem:
    """Retrieval-Augmented Generation system for document Q&A."""
    
    def __init__(self, 
                 embedding_manager,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1):
        """
        Initialize the RAG system.
        
        Args:
            embedding_manager: EmbeddingManager instance for retrieval
            model_name: Name of the LLM model to use
            temperature: Temperature for text generation
        """
        self.embedding_manager = embedding_manager
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key.strip():
            openai.api_key = api_key
            # Initialize LangChain chat model
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=api_key
            )
        else:
            # Use a simple fallback for when no API key is available
            self.llm = None
    
    def answer_question(self, 
                       question: str, 
                       top_k: int = 5,
                       include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: The question to answer
            top_k: Number of relevant chunks to retrieve
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Retrieve relevant chunks
        relevant_chunks = self.embedding_manager.search(question, top_k=top_k)
        
        if not relevant_chunks:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context from retrieved chunks
        context = self._prepare_context(relevant_chunks)
        
        # Generate answer using LLM
        answer = self._generate_answer(question, context)
        
        # Prepare response
        response = {
            'answer': answer,
            'confidence': self._calculate_confidence(relevant_chunks),
            'num_sources': len(relevant_chunks)
        }
        
        if include_sources:
            response['sources'] = self._format_sources(relevant_chunks)
        
        return response
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            title = metadata.get('title', 'Unknown')
            
            context_parts.append(f"Source {i} (from {title}):\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using the LLM or fallback method."""
        if self.llm is None:
            # Fallback: return a simple answer based on context
            return self._generate_fallback_answer(question, context)
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context. 
        Use only the information from the provided sources to answer the question. 
        If the answer cannot be found in the sources, say so clearly.
        Be concise but comprehensive in your answer.
        If you quote from the sources, indicate which source you're quoting from."""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _generate_fallback_answer(self, question: str, context: str) -> str:
        """Generate a simple answer when no LLM is available."""
        # Simple keyword-based answer generation
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Find relevant sentences from context
        sentences = context.split('. ')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_lower.split()):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Return the most relevant sentences
            answer = '. '.join(relevant_sentences[:3])  # Take first 3 relevant sentences
            if not answer.endswith('.'):
                answer += '.'
            return f"Based on the document content: {answer}"
        else:
            return "I found some relevant information in the document, but I need more context to provide a complete answer. Please try rephrasing your question or ask about specific topics mentioned in the document."
    
    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on similarity scores."""
        if not chunks:
            return 0.0
        
        # Use the highest similarity score as confidence
        max_score = max(chunk.get('similarity_score', 0) for chunk in chunks)
        
        # Normalize to 0-1 range (assuming scores are typically 0-1)
        return min(max_score, 1.0)
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for display."""
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            source = {
                'source_id': i,
                'title': metadata.get('title', 'Unknown'),
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'similarity_score': chunk.get('similarity_score', 0),
                'char_count': metadata.get('char_count', 0)
            }
            
            # Add page information if available
            if 'page_number' in metadata:
                source['page_number'] = metadata['page_number']
            
            sources.append(source)
        
        return sources
    
    def chat_with_context(self, 
                         question: str, 
                         chat_history: List[Dict[str, str]] = None,
                         top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question with optional chat history for context.
        
        Args:
            question: The current question
            chat_history: List of previous Q&A pairs
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Retrieve relevant chunks
        relevant_chunks = self.embedding_manager.search(question, top_k=top_k)
        
        if not relevant_chunks:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'confidence': 0.0,
                'chat_history': chat_history or []
            }
        
        # Prepare context
        context = self._prepare_context(relevant_chunks)
        
        # Add chat history to context if provided
        if chat_history:
            history_context = self._format_chat_history(chat_history)
            context = f"Previous conversation:\n{history_context}\n\nCurrent context:\n{context}"
        
        # Generate answer
        answer = self._generate_chat_answer(question, context, chat_history)
        
        # Update chat history
        updated_history = (chat_history or []) + [
            {'question': question, 'answer': answer}
        ]
        
        return {
            'answer': answer,
            'confidence': self._calculate_confidence(relevant_chunks),
            'sources': self._format_sources(relevant_chunks),
            'chat_history': updated_history
        }
    
    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history for context."""
        history_parts = []
        for i, exchange in enumerate(chat_history[-3:], 1):  # Last 3 exchanges
            history_parts.append(f"Q{i}: {exchange['question']}")
            history_parts.append(f"A{i}: {exchange['answer']}")
        return "\n".join(history_parts)
    
    def _generate_chat_answer(self, 
                             question: str, 
                             context: str, 
                             chat_history: List[Dict[str, str]] = None) -> str:
        """Generate answer with chat context."""
        if self.llm is None:
            # Use fallback method
            return self._generate_fallback_answer(question, context)
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context and conversation history.
        Use the provided sources to answer questions accurately.
        Consider the conversation history to provide contextually relevant answers.
        If the answer cannot be found in the sources, say so clearly.
        Be conversational and helpful."""
        
        user_prompt = f"""Context:
{context}

Current Question: {question}

Answer:"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"


def test_rag_system():
    """Test function for the RAG system."""
    from embedding_manager import EmbeddingManager
    
    # Create sample chunks
    sample_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.',
            'metadata': {'chunk_id': 0, 'title': 'ML Basics'}
        },
        {
            'text': 'Supervised learning uses labeled training data to learn a mapping from inputs to outputs.',
            'metadata': {'chunk_id': 1, 'title': 'ML Basics'}
        },
        {
            'text': 'Unsupervised learning finds hidden patterns in data without labeled examples.',
            'metadata': {'chunk_id': 2, 'title': 'ML Basics'}
        }
    ]
    
    try:
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(
            model_name="all-MiniLM-L6-v2", 
            use_openai=False
        )
        embedding_manager.create_index(sample_chunks)
        
        # Initialize RAG system
        rag = RAGSystem(embedding_manager)
        
        # Test question answering
        question = "What is the difference between supervised and unsupervised learning?"
        response = rag.answer_question(question)
        
        print(f"Question: {question}")
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']:.3f}")
        print(f"Sources: {response['num_sources']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: This test requires OpenAI API key and proper dependencies.")


if __name__ == "__main__":
    test_rag_system()
