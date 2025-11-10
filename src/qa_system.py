#!/usr/bin/env python3
"""
QA System that uses knowledge graphs to answer questions.
"""
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class QASystem:
    """Question Answering system using knowledge graphs."""

    def __init__(self, encoder=None):
        """
        Initialize QA system.

        Args:
            encoder: Optional encoder for semantic matching
        """
        self.encoder = encoder
        self.knowledge_graph = {}  # entity -> {relation -> [target_entities]}
        logger.info("Initialized QA System")

    def load_knowledge_graph(self, triplets: List[Tuple[str, str, str]]) -> None:
        """
        Load knowledge graph from triplets.

        Args:
            triplets: List of (subject, relation, object) triplets
        """
        self.knowledge_graph = {}

        for subject, relation, obj in triplets:
            if subject not in self.knowledge_graph:
                self.knowledge_graph[subject] = {}
            if relation not in self.knowledge_graph[subject]:
                self.knowledge_graph[subject][relation] = []

            if obj not in self.knowledge_graph[subject][relation]:
                self.knowledge_graph[subject][relation].append(obj)

        logger.info(f"Loaded knowledge graph with {len(triplets)} triplets")

    def load_knowledge_graph_from_file(self, file_path: Path) -> None:
        """
        Load knowledge graph from JSON file.

        Args:
            file_path: Path to JSON file containing triplets
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            triplets = [
                (
                    item.get("subject", ""),
                    item.get("relation", ""),
                    item.get("object", ""),
                )
                for item in data
            ]
        else:
            # Handle different JSON formats
            triplets = []

        self.load_knowledge_graph(triplets)

    def answer_question(
        self, question: str, context: Optional[str] = None, method: str = "word_match"
    ) -> Dict[str, Any]:
        """
        Answer a question using the knowledge graph.

        Args:
            question: The question to answer
            context: Optional context (not used in current implementation)
            method: Answer method - "word_match" or "graph_rag"

        Returns:
            Dictionary with answer, question, and supporting information
        """
        if method == "graph_rag":
            return self._answer_with_graph_rag(question)
        else:
            return self._answer_with_word_match(question)

    def _answer_with_word_match(self, question: str) -> Dict[str, Any]:
        """
        Answer question using word matching (original method).

        Args:
            question: The question to answer

        Returns:
            Dictionary with answer, question, and supporting information
        """
        # Simple pattern-based question answering
        # This is a basic implementation - can be enhanced with NLP

        question_lower = question.lower().strip()

        # Try to extract entities and relations from the question
        entities_in_graph = list(self.knowledge_graph.keys())

        # Find entities mentioned in the question
        mentioned_entities = []
        for entity in entities_in_graph:
            if entity.lower() in question_lower:
                mentioned_entities.append(entity)

        if not mentioned_entities:
            return {
                "question": question,
                "answer": "I don't have information about entities mentioned in the question.",
                "confidence": 0.0,
                "supporting_triplets": [],
                "method": "word_match",
            }

        # Try to answer based on question patterns
        answer_info = self._answer_by_pattern(question, mentioned_entities)

        if answer_info["answer"]:
            answer_info["question"] = question
            answer_info["method"] = "word_match"
            return answer_info
        else:
            # Fallback: return any information about mentioned entities
            fallback_info = self._get_entity_info(mentioned_entities[0])
            fallback_info["question"] = question
            fallback_info["method"] = "word_match"
            return fallback_info

    def _answer_by_pattern(self, question: str, entities: List[str]) -> Dict[str, Any]:
        """
        Answer question using pattern matching.

        Args:
            question: The question to answer
            entities: List of entities mentioned in the question

        Returns:
            Dictionary with answer information (without question field)
        """
        question_lower = question.lower()

        # When questions
        if "when" in question_lower and (
            "die" in question_lower
            or "death" in question_lower
            or "born" in question_lower
        ):
            for entity in entities:
                if entity in self.knowledge_graph:
                    relations = self.knowledge_graph[entity]
                    for relation in relations:
                        if any(
                            keyword in relation.lower()
                            for keyword in ["death", "die", "born", "birth", "date"]
                        ):
                            return {
                                "answer": (
                                    relations[relation][0]
                                    if relations[relation]
                                    else "Unknown"
                                ),
                                "confidence": 0.8,
                                "supporting_triplets": [
                                    (entity, relation, relations[relation][0])
                                ],
                            }

        # Who questions
        if "who" in question_lower:
            for entity in entities:
                if entity in self.knowledge_graph:
                    relations = self.knowledge_graph[entity]
                    for relation in relations:
                        if any(
                            keyword in relation.lower()
                            for keyword in [
                                "mother",
                                "father",
                                "parent",
                                "son",
                                "daughter",
                                "child",
                            ]
                        ):
                            return {
                                "answer": (
                                    relations[relation][0]
                                    if relations[relation]
                                    else "Unknown"
                                ),
                                "confidence": 0.8,
                                "supporting_triplets": [
                                    (entity, relation, relations[relation][0])
                                ],
                            }

        # What questions
        if "what" in question_lower:
            for entity in entities:
                if entity in self.knowledge_graph:
                    relations = self.knowledge_graph[entity]
                    # Return first available relation
                    for relation, objects in relations.items():
                        if objects:
                            return {
                                "answer": f"{entity} {relation} {objects[0]}",
                                "confidence": 0.6,
                                "supporting_triplets": [(entity, relation, objects[0])],
                            }

        return {"answer": "", "confidence": 0.0, "supporting_triplets": []}

    def _get_entity_info(self, entity: str) -> Dict[str, Any]:
        """Get general information about an entity."""
        if entity not in self.knowledge_graph:
            return {
                "answer": f"I don't have information about {entity}.",
                "confidence": 0.0,
                "supporting_triplets": [],
            }

        relations = self.knowledge_graph[entity]
        info_parts = []
        supporting_triplets = []

        for relation, objects in relations.items():
            for obj in objects[:3]:  # Limit to first 3 objects per relation
                info_parts.append(f"{entity} {relation} {obj}")
                supporting_triplets.append((entity, relation, obj))

        answer = (
            "; ".join(info_parts)
            if info_parts
            else f"Found entity {entity} but no detailed relations."
        )

        return {
            "answer": answer,
            "confidence": 0.5,
            "supporting_triplets": supporting_triplets,
        }

    def _answer_with_graph_rag(self, question: str) -> Dict[str, Any]:
        """
        Answer question using Graph RAG with language model.

        Args:
            question: The question to answer

        Returns:
            Dictionary with answer, question, and supporting information
        """
        try:
            # Extract relevant entities from the question
            question_lower = question.lower()
            entities_in_graph = list(self.knowledge_graph.keys())

            # Find entities mentioned in the question
            mentioned_entities = []
            for entity in entities_in_graph:
                if entity.lower() in question_lower:
                    mentioned_entities.append(entity)

            if not mentioned_entities:
                return {
                    "question": question,
                    "answer": "I don't have information about entities mentioned in the question.",
                    "confidence": 0.0,
                    "supporting_triplets": [],
                    "method": "graph_rag",
                    "graph_context": "",
                }

            # Build graph context for the language model
            graph_context = self._build_graph_context(mentioned_entities, question)

            if not graph_context.strip():
                return {
                    "question": question,
                    "answer": "I couldn't find relevant information in the knowledge graph.",
                    "confidence": 0.1,
                    "supporting_triplets": [],
                    "method": "graph_rag",
                    "graph_context": graph_context,
                }

            # Create prompt for language model
            prompt = self._create_rag_prompt(question, graph_context)

            # For now, return a structured response without actual LLM call
            # In a real implementation, you would call an LLM here
            answer = self._generate_answer_from_context(
                question, graph_context, mentioned_entities
            )

            return {
                "question": question,
                "answer": answer,
                "confidence": 0.7,
                "supporting_triplets": self._get_supporting_triplets(
                    mentioned_entities
                ),
                "method": "graph_rag",
                "graph_context": graph_context,
            }

        except Exception as e:
            logger.error(f"Error in Graph RAG answering: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "supporting_triplets": [],
                "method": "graph_rag",
                "graph_context": "",
            }

    def _build_graph_context(self, entities: List[str], question: str) -> str:
        """Build context string from knowledge graph for relevant entities."""
        context_parts = []

        for entity in entities:
            if entity in self.knowledge_graph:
                relations = self.knowledge_graph[entity]
                entity_info = f"Information about {entity}:\n"

                for relation, objects in relations.items():
                    for obj in objects[:5]:  # Limit to prevent context overflow
                        entity_info += f"- {entity} {relation} {obj}\n"

                context_parts.append(entity_info)

        return "\n".join(context_parts)

    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the language model using RAG."""
        prompt = f"""Based on the following knowledge graph information, answer the question.

Knowledge Graph Information:
{context}

Question: {question}

Please provide a concise answer based only on the information provided above. If the information doesn't contain the answer, say "I don't have enough information to answer this question."

Answer:"""
        return prompt

    def _generate_answer_from_context(
        self, question: str, context: str, entities: List[str]
    ) -> str:
        """
        Generate answer from context (simplified version without LLM).
        In a real implementation, this would call an actual language model.
        """
        question_lower = question.lower()

        # Simple rule-based answering using the context
        if "when" in question_lower and (
            "die" in question_lower or "death" in question_lower
        ):
            for entity in entities:
                if entity in self.knowledge_graph:
                    relations = self.knowledge_graph[entity]
                    for relation in relations:
                        if any(
                            keyword in relation.lower()
                            for keyword in ["death", "die", "born", "birth", "date"]
                        ):
                            for obj in relations[relation]:
                                return f"{entity} {relation} {obj}"

        if "who" in question_lower:
            for entity in entities:
                if entity in self.knowledge_graph:
                    relations = self.knowledge_graph[entity]
                    for relation in relations:
                        if any(
                            keyword in relation.lower()
                            for keyword in [
                                "mother",
                                "father",
                                "parent",
                                "son",
                                "daughter",
                            ]
                        ):
                            for obj in relations[relation]:
                                return f"{entity}'s {relation} is {obj}"

        # Fallback: return summary of available information
        return f"Based on the knowledge graph, I found information about {', '.join(entities)}."

    def _get_supporting_triplets(
        self, entities: List[str]
    ) -> List[Tuple[str, str, str]]:
        """Get all triplets for the mentioned entities."""
        supporting_triplets = []

        for entity in entities:
            if entity in self.knowledge_graph:
                relations = self.knowledge_graph[entity]
                for relation, objects in relations.items():
                    for obj in objects[:3]:  # Limit to prevent too many triplets
                        supporting_triplets.append((entity, relation, obj))

        return supporting_triplets

    def evaluate_qa(self, dataset, method: str = "graph_rag") -> Dict[str, float]:
        """
        Evaluate QA performance on a dataset.

        Args:
            dataset: Dataset with questions and ground truth answers
            method: Evaluation method - "word_match" or "graph_rag"

        Returns:
            Evaluation metrics
        """
        correct = 0
        total = 0

        for sample in dataset:
            question = sample.get("question", "")
            ground_truth = sample.get("answer", "").strip().lower()

            if not question or not ground_truth:
                continue

            result = self.answer_question(question, method=method)
            predicted = result["answer"].strip().lower()

            # Simple exact matching for evaluation
            if ground_truth in predicted or predicted in ground_truth:
                correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {"accuracy": accuracy, "correct": correct, "total": total}


if __name__ == "__main__":
    # Example usage
    qa_system = QASystem()

    # Example triplets
    example_triplets = [
        ("Lothair II", "mother", "Ermengarde of Tours"),
        ("Ermengarde of Tours", "date of death", "20 March 851"),
        ("Lothair II", "was married to", "Teutberga"),
        ("Teutberga", "date of death", "11 November 875"),
    ]

    qa_system.load_knowledge_graph(example_triplets)

    # Test questions
    questions = [
        "When did Lothair II's mother die?",
        "Who was Lothair II's mother?",
        "When did Teutberga die?",
    ]

    for question in questions:
        result = qa_system.answer_question(question)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print("---")
