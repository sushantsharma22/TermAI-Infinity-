# Reusable prompt templates for chain-of-thought, summarization, refinement, etc.

# 1) Chain of Thought
chain_of_thought_template = """\
Break down the following query into a clear chain-of-thought:
Query: {query}

Chain-of-Thought:
"""

# 2) Final Answer
final_answer_template = """\
You have the following chain-of-thought:
{reasoning}

Now, provide a concise final answer to the original query:
Query: {query}

Final Answer:
"""

# 3) Chunk Summary
chunk_summary_template = """\
Summarize the following text chunk in 1-2 sentences:
---
{chunk_text}
---
Short Summary:
"""

# 4) Combine Summaries
combine_summaries_template = """\
Below are partial summaries from different chunks of a larger text:

{partial_summaries}

Combine these partial summaries into one coherent final summary:
"""

# 5) Refinement
refinement_template = """\
You have the following text:
---
{original_text}
---
And these instructions for improvement:
{instructions}

Now provide a refined version of the text that follows these instructions:
Refined Text:
"""
