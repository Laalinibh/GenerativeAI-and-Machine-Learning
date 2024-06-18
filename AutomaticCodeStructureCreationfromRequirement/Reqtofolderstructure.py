import os
import json

# Define your folder structure
folder_structure = {
    "my-app": {
        "frontend": {
            "components": {},
            "pages": {},
            "public": {},
            "styles": {},
        },
        "backend": {
            "app": {
                "routers": {},
                "tests": {},
            },
        },
        "milvus": {},
        "llama": {},
        "web-crawler": {},
        "file-indexer": {},
    },
}


import os
import json
import openai

# Query the LLM with a prompt
openai.api_key = 'your-api-key'
response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="Create a folder structure for a full stack application with a frontend built with Next.js, a backend built with FastAPI, and integrations with Milvus, LLAMA, a web crawler, and a file indexer.",
  max_tokens=200
)

# Parse the response to get the folder structure
folder_structure = json.loads(response.choices[0].text.strip())

# Function to create the folder structure
def create_folder_structure(base_path, structure):
    for name, sub_structure in structure.items():
        path = os.path.join(base_path, name)
        os.makedirs(path, exist_ok=True)
        create_folder_structure(path, sub_structure)

# Create the folder structure
create_folder_structure(".", folder_structure)



# Write the structure to a JSON file
with open('folder_structure.json', 'w') as f:
    json.dump(folder_structure, f, indent=4)