from modules.text_generation import generate_text
from modules.prompts import chain_of_thought_template, final_answer_template

class MultiStepReasoner:
    """
    Demonstrates a multi-step chain-of-thought approach:
    1) Generate an initial reasoning (chain-of-thought).
    2) Refine that reasoning if needed.
    3) Produce a final answer.
    """

    def run_reasoning(self, query: str) -> str:
        # Step 1: Generate an initial chain-of-thought
        reasoning_prompt = chain_of_thought_template.format(query=query)
        chain_of_thought = generate_text(reasoning_prompt, max_length=250)

        # Optional: If you'd like, you can refine chain_of_thought here
        # For demonstration, we'll just produce final answer from chain_of_thought:
        final_prompt = final_answer_template.format(query=query, reasoning=chain_of_thought)
        final_answer = generate_text(final_prompt, max_length=150)

        return final_answer
