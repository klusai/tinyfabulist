import openai

# Ensure you have the OpenAI API key set up as an environment variable
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class GPTEvaluator:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

    def evaluate(self, character, trait, setting, conflict, resolution, moral, generated_fab):
        """
        Use GPT-4 to evaluate a fable based on structured input and its generated fable.

        Parameters:
        - character: The character of the fable (string).
        - trait: The trait of the character (string).
        - setting: The setting of the fable (string).
        - conflict: The conflict in the fable (string).
        - resolution: The resolution of the fable (string).
        - moral: The moral of the fable (string).
        - generated_fab: The generated fable (string).

        Returns:
        - evaluation_result: The evaluation of the fable (string).
        """
        structured_input = (
            f"Character: {character}\n"
            f"Trait: {trait}\n"
            f"Setting: {setting}\n"
            f"Conflict: {conflict}\n"
            f"Resolution: {resolution}\n"
            f"Moral: {moral}"
        )

        evaluation_prompt = (
            f"The following exercise consists of a structured input and the model's generated fable.\n"
            f"The input:\n{structured_input}\n\n"
            f"The generated fable:\n{generated_fab}\n\n"
            "Please evaluate the generated fable based on the following criteria:\n"
            "1. Grammar: Is the fable grammatically correct?\n"
            "2. Creativity: Does the fable demonstrate creative storytelling?\n"
            "3. Consistency: Does the fable align with the structured input provided?\n"
            "Also, provide a guess for the age group of the writer. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.\n\n"
            "Return your evaluation as follows:\n"
            "Grammar: x/10\nCreativity: x/10\nConsistency: x/10\nAge group: <age group guess>\n"
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a creative writer and evaluator specializing in generating and assessing fables based on structured input."
                    },
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error while generating and evaluating fable: {e}")
            return None