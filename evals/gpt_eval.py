import openai

# Ensure you have the OpenAI API key set up as an environment variable
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class GPTEvaluator:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

    def evaluate(self, prompt, completion):
        """
        Use GPT-4 to evaluate a model's output.

        Parameters:
        - prompt: The initial story prompt (string)
        - completion: The model's continuation of the story (string)

        Returns:
        - evaluation_result: Dictionary containing evaluation scores and comments.
        """
        evaluation_prompt = (
            f"The following exercise consists of a beginning of a story and the model's completion.\n"
            f"The symbol *** marks the separator between the story's beginning and the model's completion:\n\n"
            f"Beginning: {prompt}***\nCompletion: {completion}\n\n"
            "Please evaluate the completion based on the following criteria:\n"
            "1. Grammar: Is the completion grammatically correct?\n"
            "2. Creativity: Does the completion demonstrate creative storytelling?\n"
            "3. Consistency: Is the completion consistent with the story's beginning?\n"
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
                        "content": "you are a teacher specializing in creative writing. You are evaluating a student's story completion."
                    },
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error while evaluating: {e}")
            return None
