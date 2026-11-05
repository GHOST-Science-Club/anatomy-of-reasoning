In this project, we aim to analyze the reasoning process of current large language models (LLMs) with advanced reasoning capabilities, i.e., Large Reasoning Models, LRMs, based on a modified version of Alan Schoenfeld's (1985) "Episode-Timeline" framework for problem-solving. Given the model response you need to annotate the sentence-level behavior of the model response with the eight categories: Read, Analyze, Explore, Plan, Implement, Verify, Monitor, and Answer.

The [Guidebook] - [End of the Guidebook] section provides the detailed introduction and definition of each category.
The [Math Problem] - [End of the Math Problem] section provides a math problem.
The [Previous Context] - [End of the Previous Context] section provides all the previous context of the response.
The [Input] - [End of the Input] section provides the sentences that need to be annotated, in the order they appear in the response.

[Guidebook]
{guidebook}
[End of the Guidebook]

[Math Problem]
{problem}
[End of the Math Problem]

[Previous Context]
{context}
[End of the Previous Context]

[Input]
The following sentences need to be classified:
{sentences}
[End of the Input]

Each question quotes one sentence from the [Input] - [End of the Input] section. Annotate that sentence, referring to the guidebook to make the decision.
