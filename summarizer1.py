import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import pipeline
from datasets import load_dataset

# Load the dataset
ds = load_dataset("Amod/mental_health_counseling_conversations")

# Set up the summarization tool (pipeline)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Open a file to save the summaries
with open("summaries.txt", "w") as file:
    # Loop through all conversations in the dataset
    for i, entry in enumerate(ds['train']):
        context = entry['Context']
        response = entry['Response']
        
        # Combine the context and response into one piece of text
        text = f"{context} {response}"
        
        # Generate the summary
        summary = summarizer(text, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
        
        # Write the summary to the file
        file.write(f"Conversation {i + 1} Summary:\n{summary}\n\n")
        
        # Print progress to keep track
        print(f"Summarized Conversation {i + 1}/{len(ds['train'])}")

print("All conversations summarized and saved to summaries.txt!")
