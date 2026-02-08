# Day 2 Tasks Implementation

## 2. RAG System Improvements

I completed the regular tasks as well as implemented the "Additional RAG Improvements" requested in Exercise Sheet 2 by making the pipeline dynamic:

- **UI Sliders**: Added controls for **Temperature**, **Top-P**, **Top-K**, and **K Source Chunks**.

- **Dynamic Retrieval**: Changing "K Source Chunks" now instantly updates how many documents the retriever fetches.

- **Dynamic Generation**: Adjusting generation parameters (Temperature, etc.) immediately rebuilds the LLM pipeline, allowing you to experiment with model creativity and variability in real-time.

![documentation](/day2/documentation.png)
