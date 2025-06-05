import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

class WordSimilarityScorer:
    def __init__(self, model_name='glove-wiki-gigaword-300'):
        """
        Initialize the scorer with a pre-trained word embedding model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
                            Options: 'glove-wiki-gigaword-300', 'word2vec-google-news-300', 
                            'glove-twitter-200', 'fasttext-wiki-news-subwords-300'
        """
        print(f"Loading {model_name} model... This may take a moment.")
        self.model = api.load(model_name)
        print("Model loaded successfully!")
    
    def get_word_vector(self, word):
        """
        Get the vector representation of a word with fallback strategies.
        
        Args:
            word (str): The word to get vector for
            
        Returns:
            numpy.ndarray or None: Vector representation or None if word not found
        """
        word_lower = word.lower()
        
        # Strategy 1: Try the word as-is
        try:
            return self.model[word_lower]
        except KeyError:
            pass
        
        # Strategy 2: Try without hyphens (hyphenated words)
        if '-' in word_lower:
            try:
                word_no_hyphen = word_lower.replace('-', '')
                return self.model[word_no_hyphen]
            except KeyError:
                pass
        
        # Strategy 3: Try splitting hyphenated/compound words and average their vectors
        if '-' in word_lower or '_' in word_lower:
            parts = word_lower.replace('-', ' ').replace('_', ' ').split()
            if len(parts) > 1:
                vectors = []
                for part in parts:
                    try:
                        vectors.append(self.model[part])
                    except KeyError:
                        continue
                
                if vectors:
                    # Return average of component word vectors
                    return np.mean(vectors, axis=0)
        
        # Strategy 4: Try removing common suffixes/prefixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'tion', 'sion']
        for suffix in suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                try:
                    root_word = word_lower[:-len(suffix)]
                    return self.model[root_word]
                except KeyError:
                    continue
        
        print(f"Warning: Word '{word}' not found in vocabulary after trying fallback strategies")
        return None
    
    def calculate_similarity(self, word1, word2):
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            float: Similarity score between -1 and 1, or 0 if either word not found
        """
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def calculate_weighted_similarity_score(self, word_weights, target_concept, scale_range=(1, 10)):
        """
        Calculate a weighted similarity score for a group of words against a target concept.
        
        Args:
            word_weights (list): List of tuples (word, weight)
            target_concept (str): The target concept to match against
            scale_range (tuple): The output range for the final score (default: 1-10)
            
        Returns:
            dict: Contains the final score and detailed breakdown
        """
        target_vector = self.get_word_vector(target_concept)
        if target_vector is None:
            return {"error": f"Target concept '{target_concept}' not found in vocabulary"}
        
        similarities = []
        weights = []
        word_details = []
        
        total_weight = sum(weight for _, weight in word_weights)
        
        for word, weight in word_weights:
            word_vector = self.get_word_vector(word)
            if word_vector is not None:
                similarity = cosine_similarity([word_vector], [target_vector])[0][0]
                similarities.append(similarity)
                weights.append(weight)
                word_details.append({
                    'word': word,
                    'weight': weight,
                    'similarity': similarity,
                    'weighted_contribution': similarity * weight,
                    'status': 'found'
                })
            else:
                word_details.append({
                    'word': word,
                    'weight': weight,
                    'similarity': 0.0,
                    'weighted_contribution': 0.0,
                    'status': 'not_found'
                })
        
        if not similarities:
            return {"error": "No valid words found in vocabulary"}
        
        # Calculate weighted average similarity
        weighted_sum = sum(sim * weight for sim, weight in zip(similarities, weights))
        used_weight_sum = sum(weights)
        weighted_avg_similarity = weighted_sum / used_weight_sum if used_weight_sum > 0 else 0
        
        # Transform similarity from [-1, 1] to [0, 1] range
        normalized_similarity = (weighted_avg_similarity + 1) / 2
        
        # Scale to desired range
        min_scale, max_scale = scale_range
        final_score = min_scale + (normalized_similarity * (max_scale - min_scale))
        
        return {
            'final_score': round(final_score, 2),
            'raw_weighted_similarity': round(weighted_avg_similarity, 4),
            'normalized_similarity': round(normalized_similarity, 4),
            'target_concept': target_concept,
            'total_words_processed': len([w for w in word_details if w['similarity'] != 0]),
            'total_words_input': len(word_weights),
            'word_details': word_details
        }
    
    def print_detailed_results(self, results):
        """
        Print a detailed breakdown of the similarity calculation results.
        
        Args:
            results (dict): Results from calculate_weighted_similarity_score
        """
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"\n=== Similarity Analysis for '{results['target_concept']}' ===")
        print(f"Final Score: {results['final_score']}/10")
        print(f"Raw Weighted Similarity: {results['raw_weighted_similarity']}")
        print(f"Words Processed: {results['total_words_processed']}/{results['total_words_input']}")
        
        print(f"\nWord-by-word breakdown:")
        print(f"{'Word':<20} {'Weight':<8} {'Similarity':<10} {'Contribution':<12} {'Status':<10}")
        print("-" * 70)
        
        for detail in results['word_details']:
            status = detail.get('status', 'found')
            print(f"{detail['word']:<20} {detail['weight']:<8.4f} {detail['similarity']:<10.4f} {detail['weighted_contribution']:<12.4f} {status:<10}")


# Example usage
if __name__ == "__main__":
    # Initialize the scorer with GloVe (often better for compound words)
    # You can also try 'fasttext-wiki-news-subwords-300' for better handling of rare words
    scorer = WordSimilarityScorer('glove-wiki-gigaword-300')
    
    # Example word list with weights (including problematic compound words)
    word_weights = [
        ('spacesuit', 0.27710309624671936),
        ('astronaut', 0.2649446725845337),
        ('martian', 0.256915807723999),
        ('horseback', 0.2514975965023041),
        ('horse', 0.24484850466251373),
        ('desert', 0.22846719622612),
        ('riding', 0.2215965986251831),
        ('image', 0.21975240111351013),
        ('medium-brightness', 0.21675395965576172),  # This will be handled by fallback strategies
        ('helmet', 0.21443219482898712)
    ]
    
    print("Testing fallback strategies for compound words:")
    print(f"medium-brightness found: {scorer.get_word_vector('medium-brightness') is not None}")
    print(f"horseback found: {scorer.get_word_vector('horseback') is not None}")
    
    # Calculate similarity to "happiness"
    results = scorer.calculate_weighted_similarity_score(word_weights, "happiness")
    
    # Print detailed results
    scorer.print_detailed_results(results)
    
    # You can also test with other concepts
    print("\n" + "="*70)
    results2 = scorer.calculate_weighted_similarity_score(word_weights, "Lonely")
    scorer.print_detailed_results(results2)
    
    print("\n" + "="*70)
    results3 = scorer.calculate_weighted_similarity_score(word_weights, "space")
    scorer.print_detailed_results(results3)

    from analysis_model import ImageTextMatchingModel
    model = ImageTextMatchingModel()
    image_path = "../docs/astronaut_rides_horse.png"
    score_of_clip = model.score_attributes(image_path, ["happiness", "Lonely", "space"])
    print(score_of_clip)