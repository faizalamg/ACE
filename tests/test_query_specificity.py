"""Test script for query specificity analysis."""
from ace.retrieval_optimized import QuerySpecificityScorer

scorer = QuerySpecificityScorer()

# Test queries including 'something is broken'
test_queries = [
    'something is broken',
    'How to debug timeout errors in production?',
    'prevent SQL injection',
    'authentication',
    'memory leak in React hooks',
    'help',
    'fix bug',
    'pytest async fixtures not working with database connection pooling'
]

print('Query Specificity Analysis')
print('=' * 80)
for q in test_queries:
    score = scorer.score(q)
    print(f'Query: "{q}"')
    print(f'  Word Count: {score.word_count}')
    print(f'  Specificity: {score.specificity_score}')
    print(f'  Expansion Level: {score.expansion_level}')
    print(f'  Use LLM: {score.use_llm_expansion}')
    print(f'  Use Structured: {score.use_structured_expansion}')
    print(f'  Terms Limit: {score.expansion_terms_limit}')
    print(f'  Rationale: {score.rationale}')
    print()
