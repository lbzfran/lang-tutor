
# Introduction
The goal is to develop a simple language tutor that can reliably train
a user in vocabulary, grammar, and usage.


# Notes
## 6/7/25
The test case right now is English-to-Korean.

Now that the RAG and Model are coded in, I can see how badly the RAG part
sucks in terms of data. The context it's able to pull is not relevant enough
to make a quiz out of it. The model currently being used is also really bad
(it's a quantized DeepSeek model).

My next steps right now are:
- [ ] look for/organize better data
- [ ] look for a better suited, low-powered model (if possible)
- [ ] improve on prompting templates
- [ ] integrate quiz results interactively
