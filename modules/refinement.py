from modules.text_generation import generate_text
from modules.prompts import refinement_template

class Refiner:
    """
    Runs a second-pass refinement of an existing text using instructions.
    """

    def refine_text(self, text: str, instructions: str) -> str:
        prompt = refinement_template.format(
            original_text=text,
            instructions=instructions
        )
        refined = generate_text(prompt, max_length=200)
        return refined
