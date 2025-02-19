import statistics

from src.utils.data_manager import DataManager

class EvaluationMetrics:
    def __init__(self, diversity_filename='diversity_evaluation.json', evaluation_filename='evaluation_results.json'):
        self._diversity_filename = diversity_filename
        self._evaluation_filename = evaluation_filename
        self._average_scores = self.get_average_scores()

    def get_average_scores(self):
        """
        Calculates and prints the average grammar, creativity, and consistency scores
        from a JSON file.
        """
        filename = self._evaluation_filename
        entries = DataManager.read_json_file(filename)

        grammar_scores = []
        creativity_scores = []
        consistency_scores = []

        for entry in entries:
            try:
                grammar_scores.append(int(entry['grammar'].split('/')[0]))
                creativity_scores.append(int(entry['creativity'].split('/')[0]))
                consistency_scores.append(int(entry['consistency'].split('/')[0]))
            except (KeyError, ValueError, IndexError) as e:
                print(f"Error processing entry: {entry}. Skipping. Error: {e}")
                continue

        if not grammar_scores or not creativity_scores or not consistency_scores:
            print("Error: No valid scores found in the entries.")
            return

        avg_creativity = statistics.mean(creativity_scores)
        avg_grammar = statistics.mean(grammar_scores)
        avg_consistency = statistics.mean(consistency_scores)

        return {
            "creativity": float(f"{avg_creativity:.2f}"),
            "grammar": float(f"{avg_grammar:.2f}"),
            "consistency": float(f"{avg_consistency:.2f}")
        }

    @property
    def creativity(self):
        return self._average_scores['creativity']

    @property
    def grammar(self):
        return self._average_scores['grammar']

    @property
    def consistency(self):
        return self._average_scores['consistency']

    @property
    def diversity_score(self):
        """
        Retrieves and returns the diversity score from a JSON file.

        Returns:
            int: The diversity score, scaled by a factor of 10.
                 Returns None if the score is not found or invalid.
        """
        try:
            entries = DataManager.read_json_file(self._diversity_filename)
            diversity_score = int(float(entries['diversity_evaluation']['diversity_score']) * 10)
            return diversity_score
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error retrieving diversity score from {self._diversity_filename}: {e}")
            return None
