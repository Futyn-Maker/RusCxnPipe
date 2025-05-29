"""Using ConstructionClassifier component separately."""

from ruscxnpipe import ConstructionClassifier

def main():
    # Initialize classifier
    classifier = ConstructionClassifier(
        model_name="Futyn-Maker/ruscxn-classifier",
        threshold=0.5
    )
    
    # Prepare queries and candidates
    queries = ["Таня танцевала без устали, танцевала со всеми подряд."]
    
    # Candidates that we want to classify (could come from semantic search)
    candidates_list = [[
        {"id": "pattern_1", "pattern": "VP без устали"},
        {"id": "pattern_2", "pattern": "VP всё/все подряд"}, 
        {"id": "pattern_3", "pattern": "VP кто где"},  # This shouldn't be present
        {"id": "pattern_4", "pattern": "что NP-Dat Cop нужно/надо?"}  # This shouldn't be present
    ]]
    
    # Classify candidates
    results = classifier.classify_candidates(
        queries=queries,
        candidates_list=candidates_list,
        batch_size=16
    )
    
    print("=== Classification Results ===")
    print(f"Query: {queries[0]}")
    print("\nClassification results:")
    
    for candidate in results[0]:
        status = "PRESENT" if candidate['is_present'] == 1 else "NOT PRESENT"
        print(f"  - Pattern: {candidate['pattern']}")
        print(f"    Status: {status}")
        print(f"    ID: {candidate['id']}")
    
    # Show only positive classifications
    positive_candidates = [c for c in results[0] if c['is_present'] == 1]
    print(f"\nPositive predictions: {len(positive_candidates)} out of {len(results[0])}")
    
    # Get model information
    model_info = classifier.get_model_info()
    print(f"\n=== Model Information ===")
    print(f"Model: {model_info['model_name']}")
    print(f"Device: {model_info['device']}")
    print(f"Threshold: {model_info['threshold']}")

if __name__ == "__main__":
    main()
