from simpli_agent import Agent

agent = Agent(semantic_memory=True)
agent.run('The capital of France is Paris')
agent.run('The capital of Japan is Tokyo')
agent.run('The capital of Germany is Berlin')

results = agent.search_memory('capital of France', semantic=True)
print('Semantic search results:', len(results))
for r in results:
    score = r.get('score', 'N/A')
    content = r['content'][:50]
    print('  Score:', score, '-', content)

results = agent.search_memory('Tokyo', semantic=False)
print('Keyword search results:', len(results))