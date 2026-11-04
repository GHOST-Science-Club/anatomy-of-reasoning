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
The following sentences which you need to classify:
{sentences}
[End of the Input]

[Format]
You should format the output in json format regarding the index, a short rationale and the class of the indexed sentence. The class must be one of: Read, Analyze, Plan, Implement, Explore, Verify, Monitor, Answer. The format is as follows:
{
  "sentences": [
    {"index": 1, "reason": "The short reason of the classification", "category": "Read"}
  ]
}
You should strictly follow the index number of the sentence in the [Input] - [End of the Input] section.
[End of the Format]

Now, annotate the sentences in the [Input] - [End of the Input] section. Refer to the guidebook to make the decision. Strictly follow the index number of the sentence in the [Input] - [End of the Input] section for labeling. You should output the label for {n} sentences.
