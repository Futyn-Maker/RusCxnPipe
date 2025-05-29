"""Using SemanticSearch component separately."""

from ruscxnpipe import SemanticSearch

def main():
    # Initialize semantic search
    search = SemanticSearch(
        model_name="Futyn-Maker/ruscxn-embedder"
    )
    
    # Find candidates for multiple queries
    queries = [
        "Что вам здесь нужно?",
        "Мои друзья разъехались и исчезли кто где."
    ]
    
    results = search.find_candidates(
        queries=queries,
        n=5  # Get top 5 candidates for each query
    )
    
    print("=== Semantic Search Results ===")
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {result['query']}")
        print(f"Top candidates:")
        
        for j, candidate in enumerate(result['candidates'], 1):
            print(f"  {j}. Pattern: {candidate['pattern']}")
            print(f"     Similarity: {candidate['similarity']:.3f}")
            print(f"     ID: {candidate['id']}")
    
    # Get cache information
    cache_info = search.get_cache_info()
    print(f"\n=== Cache Information ===")
    print(f"Cache size: {cache_info['cache_size']} patterns")
    print(f"Cache file: {cache_info['cache_file']}")

if __name__ == "__main__":
    main()
