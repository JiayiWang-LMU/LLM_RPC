# LLM RPC

This project helps users compare the quality of expected results from Large Language Models (LLMs) using only a prompt. By running the `generate_ranking.py` script and inputting a prompt, the system will output a ranking of LLMs that provide the best answers according to the **MT-bench dataset**, which is based on human experts' judgments.

## Examples

## How It Works
1. **Run the Script**: Execute the `generate_ranking.py` script.
2. **Input a Prompt**: Provide a prompt when prompted by the system.
3. **Get Rankings**: The system will output a ranking of LLMs based on the quality of their responses.

**Prompt**: Explain antibiotics  
**LLM ranking**:  
claude-v1 > gpt-4 > gpt-3.5-turbo > vicuna-13b-v1.2 > alpaca-13b > llama-13b  

**Prompt**: Plan a trip to Hawaii for me  
**LLM ranking**:  
gpt-4 > claude-v1 > gpt-3.5-turbo > vicuna-13b-v1.2 > alpaca-13b > llama-13b  

**Prompt**: You have been tasked with designing a solar-powered water heating system for a residential building. Describe the key components and considerations you would include in your design. Design a five-step workflow.  
**LLM ranking**:  
claude-v1 > gpt-4 > vicuna-13b-v1.2 > gpt-3.5-turbo > alpaca-13b > llama-13b  
