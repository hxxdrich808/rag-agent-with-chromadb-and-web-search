# Add missing dependency for vectorstore module

## Requirements
- [high] Include langchain-text-splitters in requirements.txt: Add the line `langchain-text-splitters==0.1.0` (or latest compatible version) to the `requirements.txt` file so that the module can import and use `RecursiveCharacterTextSplitter`. Ensure the package is installed during environment setup and that no other dependency conflicts arise.
