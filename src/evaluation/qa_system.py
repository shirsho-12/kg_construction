#!/usr/bin/env python3
"""
QA System that uses knowledge graphs to answer questions with LLM-based Graph RAG.
"""
import textwrap
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from pathlib import Path
import json
from evaluation.faiss_index import FaissIndex
from utils import logger
import networkx as nx
from utils import logger


class QASystem:
    """Question Answering system using knowledge graphs with LLM-based Graph RAG."""

    def __init__(self, encoder, prompts_dir: Optional[Path] = None, k_hop: int = 0):
        """
        Initialize QA system.

        Args:
            encoder: Encoder for LLM generation
            prompts_dir: Directory containing prompt templates
            k_hop: Number of hops for graph context expansion
        """
        self.encoder = encoder
        self.knowledge_graph = defaultdict(lambda: nx.MultiDiGraph())
        self.prompts_dir = (
            prompts_dir or Path(__file__).parent.parent.parent / "prompts"
        )
        self.triplets = defaultdict(list)
        self.faiss_indices = {}  # Store FAISS index per question ID
        self.passages_cache = {}  # Store passages per question ID
        self.k_hop = k_hop
        self._load_prompt_templates()
        logger.info("Initialized QA System with LLM-based Graph RAG")

    def load_knowledge_graph(
        self, triplets: Dict[str, List[Tuple[str, str, str]]]
    ) -> None:
        """
        Load knowledge graph from triplets.

        Args:
            triplets: List of (subject, relation, object) triplets
        """
        for entity_id, triplet_list in triplets.items():
            for subject, relation, obj in triplet_list:
                self.knowledge_graph[entity_id].add_edge(
                    subject, obj, relation=relation
                )
        logger.info(f"Loaded knowledge graph with {len(triplets)} triplets")

    def load_knowledge_graph_from_file(self, file_path: Path) -> None:
        """
        Load knowledge graph from the triplets.txt JSON file.

        Args:
            file_path: Path to folder containing the previous step's outputs
        """
        with open(file_path / "triplets.txt", "r", encoding="utf-8") as f:
            data = json.load(f)
        triplets = {}
        if isinstance(data, list):
            triplets["0"] = []
            for obj in data:
                if isinstance(obj, list) and len(obj) == 3:
                    triplets["0"].append((obj[0], obj[1], obj[2]))
                elif isinstance(obj, dict):
                    triplets["0"].append(
                        (
                            obj.get("subject", ""),
                            obj.get("relation", ""),
                            obj.get("object", ""),
                        )
                    )
                    self.triplets["0"].append(
                        (
                            obj.get("subject", ""),
                            obj.get("relation", ""),
                            obj.get("object", ""),
                        )
                    )
        elif isinstance(data, dict):
            for key, objs in data.items():
                triplets[key] = []
                for obj in objs:
                    if isinstance(obj, list) and len(obj) == 3:
                        triplets[key].append((obj[0], obj[1], obj[2]))
                    elif isinstance(obj, dict):
                        triplets[key].append(
                            (
                                obj.get("subject", ""),
                                obj.get("relation", ""),
                                obj.get("object", ""),
                            )
                        )
            self.triplets.update(triplets)
        return self.load_knowledge_graph(triplets)

    def load_graph_for_passage(self, q_id: str, triplets_data: List[Tuple[str, str, str]]) -> None:
        """
        Load knowledge graph for a specific question/passage ID.
        
        Args:
            q_id: Question/passage ID
            triplets_data: List of (subject, relation, object) triplets for this question
        """
        # Clear existing data for this ID
        self.knowledge_graph[q_id] = nx.MultiDiGraph()
        self.triplets[q_id] = []
        
        # Load triplets into graph and cache
        for i, (subject, relation, obj) in enumerate(triplets_data):
            self.knowledge_graph[q_id].add_edge(subject, obj, relation=relation, triplet_id=i)
            self.triplets[q_id].append({
                "head": subject,
                "relation": relation, 
                "tail": obj
            })
        
        logger.info(f"Loaded {len(triplets_data)} triplets for q_id={q_id}")

    def triplet_to_sentence(self, triplet: Dict[str, str]) -> str:
        # simple natural language rendering
        return f"{triplet['head']} {triplet['relation']} {triplet['tail']}."

    def build_passages(
        self,
        graph: nx.MultiDiGraph,
        triplets: List[Dict[str, str]],
        k_hop: int = 0,
        max_context_triples: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Returns list of dicts:
            {'id': idx, 'text': sentence, 'meta': {...}}
        If k_hop > 0, include neighboring triplets up to k hops (naive BFS by nodes).
        """
        passages = []
        for i, tr in enumerate(triplets):
            base = self.triplet_to_sentence(tr)
            context_texts = []
            if k_hop > 0:
                # gather neighbors around head and tail
                seeds = [tr["head"], tr["tail"]]
                seen_trip_ids = set([i])
                # BFS up to k_hop in terms of nodes
                frontier = set(seeds)
                for depth in range(k_hop):
                    next_frontier = set()
                    for node in frontier:
                        # outgoing edges
                        for u, v, key, data in graph.out_edges(
                            node, keys=True, data=True
                        ):
                            tid = data.get("triplet_id", key)
                            if tid not in seen_trip_ids:
                                seen_trip_ids.add(tid)
                                ttrip = triplets[tid]
                                context_texts.append(self.triplet_to_sentence(ttrip))
                                next_frontier.add(v)
                        # incoming edges
                        for u, v, key, data in graph.in_edges(
                            node, keys=True, data=True
                        ):
                            tid = data.get("triplet_id", key)
                            if tid not in seen_trip_ids:
                                seen_trip_ids.add(tid)
                                ttrip = triplets[tid]
                                context_texts.append(self.triplet_to_sentence(ttrip))
                                next_frontier.add(u)
                    frontier = next_frontier
                    if len(context_texts) >= max_context_triples:
                        break
            if context_texts:
                # keep short
                context_texts = context_texts[:max_context_triples]
                full = base + " Context: " + " ".join(context_texts)
            else:
                full = base
            passages.append(
                {
                    "id": i,
                    "text": full,
                    "meta": {
                        "head": tr["head"],
                        "relation": tr["relation"],
                        "tail": tr["tail"],
                    },
                }
            )
        return passages

    def build(self, q_id: str, normalize: bool = True):
        """Build FAISS index and cache passages for a specific question ID."""
        graph = self.knowledge_graph[q_id]
        triplets = self.triplets[q_id]
        passages = self.build_passages(graph, triplets, k_hop=self.k_hop)
        texts = [p["text"] for p in passages]
        logger.info(
            f"[build] embedding {len(texts)} passages for q_id={q_id} with model (dim={self.encoder.embedding_dim}) ..."
        )
        
        if not texts:
            logger.warning(f"No passages found for q_id={q_id}")
            self.faiss_indices[q_id] = None
            self.passages_cache[q_id] = []
            return passages
            
        vectors = self.encoder.encode(texts)
        faiss_index = FaissIndex(
            embedding_dim=self.encoder.embedding_dim, normalize=normalize
        )
        ids = [p["id"] for p in passages]
        faiss_index.add(vectors, ids)
        
        # Store per question ID
        self.faiss_indices[q_id] = faiss_index
        self.passages_cache[q_id] = passages
        
        logger.info(f"[build] index built for q_id={q_id}.")
        return passages

    def retrieve(self, query: str, q_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve passages for a specific question ID."""
        if q_id not in self.faiss_indices or self.faiss_indices[q_id] is None:
            logger.warning(f"No FAISS index found for q_id={q_id}")
            return []
            
        passages = self.passages_cache.get(q_id, [])
        if not passages:
            logger.warning(f"No passages cached for q_id={q_id}")
            return []
            
        qvec = self.encoder.encode([query])
        results = self.faiss_indices[q_id].search(qvec, top_k=top_k)
        out = []
        for pid, score in results:
            if pid < len(passages):
                p = passages[pid]
                out.append(
                    {
                        "passage_id": pid,
                        "score": score,
                        "text": p["text"],
                        "meta": p["meta"],
                    }
                )
        return out

    def answer_with_transformers(
        self,
        question: str,
        retrieved: List[Dict[str, Any]],
        q_id: str = "0",
    ) -> str:
        """Generate answer using retrieved passages for a specific question ID."""
        context = "\n".join([f"[{i}] {r['text']}" for i, r in enumerate(retrieved)])
        prompt = textwrap.dedent(
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely."
        )
        out = self.encoder.generate_completion([{"role": "user", "content": prompt}])
        return out[0].strip()

    def answer_question(self, question: str, q_id: str = "0", top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using the knowledge graph with LLM-based Graph RAG.

        Args:
            question: The question to answer
            q_id: Question/passage ID to identify which graph and index to use
            top_k: Number of top passages to retrieve

        Returns:
            Dictionary with answer, question, and supporting information
        """
        try:
            # Ensure we have built the index for this question ID
            if q_id not in self.faiss_indices:
                logger.info(f"Building index for q_id={q_id}")
                self.build(q_id)
            
            # Retrieve relevant passages using FAISS
            retrieved_passages = self.retrieve(question, q_id, top_k=top_k)
            
            if not retrieved_passages:
                return {
                    "question": question,
                    "answer": "I couldn't find relevant information in the knowledge graph.",
                    "confidence": 0.1,
                    "supporting_triplets": [],
                    "retrieved_passages": [],
                }

            # Generate answer using LLM with retrieved passages
            answer = self.answer_with_transformers(question, retrieved_passages, q_id=q_id)

            return {
                "question": question,
                "answer": answer,
                "confidence": 0.8,
                "retrieved_passages": retrieved_passages,
                "q_id": q_id,
            }

        except Exception as e:
            logger.error(f"Error in Graph RAG answering for q_id={q_id}: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "retrieved_passages": [],
                "q_id": q_id,
            }

    def _load_prompt_templates(self):
        """Load prompt templates from files."""
        try:
            qa_prompt_path = self.prompts_dir / "qa_prompt.txt"
            qa_example_path = self.prompts_dir / "qa_example.txt"

            with open(qa_prompt_path, "r", encoding="utf-8") as f:
                self.qa_prompt_template = f.read().strip()

            with open(qa_example_path, "r", encoding="utf-8") as f:
                self.qa_examples = f.read().strip()

            logger.info("Loaded prompt templates successfully")
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
            # Fallback to basic prompt
            self.qa_prompt_template = "Based on the following information, answer the question: {question}\n\nInformation: {context}\n\nAnswer:"
            self.qa_examples = ""

    def _generate_llm_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with prompt template.

        Args:
            question: The user's question
            context: Graph context information

        Returns:
            Generated answer as a complete sentence
        """
        # Format the prompt with examples
        full_prompt = (
            self.qa_examples
            + "\n\n"
            + self.qa_prompt_template.format(
                context=context.strip(), question=question.strip()
            )
        )

        try:
            # Generate response using encoder
            messages = [{"role": "user", "content": full_prompt}]
            responses = self.encoder.generate_completion(
                messages, max_length=200, answer_prefix=""
            )

            if responses and len(responses) > 0:
                answer = responses[0].strip()
                # Ensure it's a complete sentence and ends with proper punctuation
                if answer and not answer.endswith((".", "!", "?")):
                    answer += "."
                return answer
            else:
                return ""

        except Exception as e:
            logger.error(f"Error generating LLM answer: {e}")
            return "ERROR"

    def evaluate_qa(self, dataset) -> Dict[str, float]:
        """
        Evaluate QA performance on a dataset.

        Args:
            dataset: Dataset with questions and ground truth answers

        Returns:
            Evaluation metrics
        """
        correct = 0
        total = 0

        for sample in dataset:
            question = sample.get("question", "")
            ground_truth = sample.get("answer", "").strip().lower()
            q_id = sample.get("id", str(total))  # Use sample ID or fallback to index

            if not question or not ground_truth:
                continue

            # Build graph and index for this specific question
            logger.info(f"Evaluating question {total + 1}/{len(dataset)} (q_id={q_id})")
            result = self.answer_question(question, q_id=q_id)
            predicted = result["answer"].strip().lower()

            # Simple exact matching for evaluation
            if ground_truth in predicted or predicted in ground_truth:
                correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {"accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_qa_with_graph_construction(self, dataset, triplets_per_question: Dict[str, List[Tuple[str, str, str]]]) -> Dict[str, float]:
        """
        Evaluate QA performance with graph construction for each question.
        
        Args:
            dataset: Dataset with questions and ground truth answers
            triplets_per_question: Dictionary mapping question IDs to their triplets
            
        Returns:
            Evaluation metrics
        """
        correct = 0
        total = 0
        
        for sample in dataset:
            question = sample.get("question", "")
            ground_truth = sample.get("answer", "").strip().lower()
            q_id = sample.get("id", str(total))  # Use sample ID or fallback to index
            
            if not question or not ground_truth:
                continue
                
            # Load graph for this specific question
            if q_id in triplets_per_question:
                self.load_graph_for_passage(q_id, triplets_per_question[q_id])
            else:
                logger.warning(f"No triplets found for q_id={q_id}")
                continue
            
            # Build graph and index for this specific question
            logger.info(f"Evaluating question {total + 1}/{len(dataset)} (q_id={q_id})")
            result = self.answer_question(question, q_id=q_id)
            predicted = result["answer"].strip().lower()
            
            # Simple exact matching for evaluation
            if ground_truth in predicted or predicted in ground_truth:
                correct += 1
                
            total += 1
            
        accuracy = correct / total if total > 0 else 0.0
        
        return {"accuracy": accuracy, "correct": correct, "total": total}
