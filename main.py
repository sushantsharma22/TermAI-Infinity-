#!/usr/bin/env python3
import argparse
import sys

from modules.config import check_environment
from modules.text_generation import generate_text
from modules.rag_pipeline import RAGPipeline
from modules.reasoning import MultiStepReasoner
from modules.summarization import Summarizer
from modules.refinement import Refiner

def main():
    """
    Main CLI entry for TermAI Infinity (Ultra Advanced Edition).
    """

    parser = argparse.ArgumentParser(
        description="TermAI Infinity: Offline advanced LLM toolkit with retrieval, summarization, refinement, etc."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Subcommand: generate
    gen_parser = subparsers.add_parser("generate", help="Generate text from a local LLM")
    gen_parser.add_argument("--prompt", required=True, help="Prompt for text generation")
    gen_parser.add_argument("--max_length", type=int, default=100, help="Maximum tokens in output")
    gen_parser.add_argument("--refine", action="store_true", help="If true, run iterative refinement after generation")

    # Subcommand: rag
    rag_parser = subparsers.add_parser("rag", help="Question-answering with retrieval-augmented generation")
    rag_parser.add_argument("--question", required=True, help="Question to be answered using local docs")

    # Subcommand: reason
    reason_parser = subparsers.add_parser("reason", help="Multi-step chain-of-thought reasoning")
    reason_parser.add_argument("--query", required=True, help="Complex query to reason about")

    # Subcommand: summarize
    summarize_parser = subparsers.add_parser("summarize", help="Chunk-based summarization of a text file")
    summarize_parser.add_argument("--file", required=True, help="Path to the text file to summarize")

    # Subcommand: refine
    refine_parser = subparsers.add_parser("refine", help="Refine or improve an existing text snippet")
    refine_parser.add_argument("--text", required=True, help="The text to refine")
    refine_parser.add_argument("--instructions", required=True, help="Refinement instructions")

    args = parser.parse_args()

    # Basic environment checks
    check_environment()

    if args.command == "generate":
        do_generate(args.prompt, args.max_length, refine=args.refine)
    elif args.command == "rag":
        do_rag(args.question)
    elif args.command == "reason":
        do_reason(args.query)
    elif args.command == "summarize":
        do_summarize(args.file)
    elif args.command == "refine":
        do_refine(args.text, args.instructions)
    else:
        parser.print_help()

def do_generate(prompt, max_length=100, refine=False):
    """
    Generate text from the local LLM. Optionally refine the result after generation.
    """
    output = generate_text(prompt, max_length=max_length)
    print("\n=== GENERATED TEXT ===")
    print(output)

    if refine:
        refiner = Refiner()
        refined_output = refiner.refine_text(output, instructions="Improve style and clarity.")
        print("\n=== REFINED TEXT ===")
        print(refined_output)

def do_rag(question):
    """
    Perform retrieval-augmented Q&A from local docs.
    """
    rag = RAGPipeline()
    answer = rag.ask_question(question)
    print("\n=== RAG ANSWER ===")
    print(answer)

def do_reason(query):
    """
    Perform multi-step reasoning.
    """
    reasoner = MultiStepReasoner()
    final_answer = reasoner.run_reasoning(query)
    print("\n=== REASONED ANSWER ===")
    print(final_answer)

def do_summarize(filepath):
    """
    Summarize a large file with chunk-based approach.
    """
    summ = Summarizer()
    final_summary = summ.summarize_file(filepath)
    print("\n=== SUMMARY ===")
    print(final_summary)

def do_refine(text, instructions):
    """
    Refine/improve text based on user instructions.
    """
    refiner = Refiner()
    improved_text = refiner.refine_text(text, instructions=instructions)
    print("\n=== REFINED TEXT ===")
    print(improved_text)

if __name__ == "__main__":
    main()
