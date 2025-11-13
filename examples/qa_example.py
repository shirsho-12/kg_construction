#!/usr/bin/env python3
"""
Example usage of the LLM-based QA system with different encoder backends.
"""

from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.qa_system import QASystem
from core.encoder import Encoder


def example_with_transformers():
    """Example using transformers-based encoder."""
    print("=== QA System with Transformers Encoder ===")

    try:
        from config import BASE_ENCODER_MODEL

        # Initialize encoder with transformers
        encoder = Encoder(
            model_name_or_path=BASE_ENCODER_MODEL, framework="transformers"
        )

        # Initialize QA system
        qa_system = QASystem(encoder=encoder)

        # Load example knowledge graph
        example_triplets = [
            ("Albert Einstein", "was born in", "1879"),
            ("Albert Einstein", "developed", "theory of relativity"),
            ("Albert Einstein", "won", "Nobel Prize in Physics"),
            ("Marie Curie", "discovered", "polonium"),
            ("Marie Curie", "discovered", "radium"),
            ("Marie Curie", "won", "Nobel Prize in Chemistry"),
            ("Leonardo da Vinci", "painted", "Mona Lisa"),
            ("Leonardo da Vinci", "painted", "Last Supper"),
        ]

        qa_system.load_knowledge_graph(example_triplets)

        # Test questions
        questions = [
            "When was Albert Einstein born?",
            "What did Marie Curie discover?",
            "What did Leonardo da Vinci paint?",
        ]

        for question in questions:
            result = qa_system.answer_question(question)
            print(f"Question: {question}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']}")
            print("---")

    except Exception as e:
        print(f"Error with transformers example: {e}")


def example_with_openai():
    """Example using OpenAI encoder."""
    print("\n=== QA System with OpenAI Encoder ===")

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable to test."
        )
        return

    try:
        # Initialize encoder with OpenAI
        encoder = Encoder(model_name_or_path="gpt-3.5-turbo", framework="openai")

        # Initialize QA system
        qa_system = QASystem(encoder=encoder)

        # Load example knowledge graph
        example_triplets = [
            ("Albert Einstein", "was born in", "1879"),
            ("Albert Einstein", "developed", "theory of relativity"),
            ("Albert Einstein", "won", "Nobel Prize in Physics"),
            ("Marie Curie", "discovered", "polonium"),
            ("Marie Curie", "discovered", "radium"),
            ("Marie Curie", "won", "Nobel Prize in Chemistry"),
        ]

        qa_system.load_knowledge_graph(example_triplets)

        # Test questions
        questions = [
            "When was Albert Einstein born?",
            "What did Marie Curie discover?",
        ]

        for question in questions:
            result = qa_system.answer_question(question)
            print(f"Question: {question}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']}")
            print("---")

    except Exception as e:
        print(f"Error with OpenAI example: {e}")


def example_with_openrouter():
    """Example using OpenRouter encoder."""
    print("\n=== QA System with OpenRouter Encoder ===")

    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable to test."
        )
        return

    try:
        # Initialize encoder with OpenRouter
        encoder = Encoder(
            model_name_or_path="meta-llama/llama-3.2-3b-instruct:free",
            framework="openrouter",
        )

        # Initialize QA system
        qa_system = QASystem(encoder=encoder)

        # Load example knowledge graph
        example_triplets = [
            ("Albert Einstein", "was born in", "1879"),
            ("Albert Einstein", "developed", "theory of relativity"),
            ("Albert Einstein", "won", "Nobel Prize in Physics"),
        ]

        qa_system.load_knowledge_graph(example_triplets)

        # Test question
        question = "When was Albert Einstein born?"
        result = qa_system.answer_question(question)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")

    except Exception as e:
        print(f"Error with OpenRouter example: {e}")


if __name__ == "__main__":
    print("QA System Examples")
    print("==================")

    # Run examples
    example_with_transformers()
    example_with_openai()
    example_with_openrouter()

    print("\n=== Usage Instructions ===")
    print("1. For transformers: No API key needed, uses local models")
    print("2. For OpenAI: Set OPENAI_API_KEY environment variable")
    print("3. For OpenRouter: Set OPENROUTER_API_KEY environment variable")
    print("4. The QA system automatically loads prompts from the prompts/ directory")
