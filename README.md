# perplexity-deep-search

Deep Research, inspired by Perplexity, is an automated research assistant designed to streamline the process of gathering, analyzing, and synthesizing information from the web. It leverages LLMs and search engines to generate search queries, retrieve relevant results, and produce structured markdown reports.

## Features
- **Automated Query Generation**: Uses LLMs to generate and refine search queries based on user input.
- **Web Search Integration**: Retrieves relevant results from search engines and APIs.
- **Structured Reports**: Produces well-organized markdown reports tailored to the user's query.
- **Streamlit UI**: Provides an interactive interface for managing research workflows.

## File Descriptions
- **main.py**: The entry point for the Streamlit app, orchestrating the research workflow and UI.
- **langgraph_workflow.py**: Defines the state-driven workflow for query generation, refinement, and report generation.
- **prompts.py**: Stores prompt templates for generating search queries, refining them, and creating final reports.
- **utils.py**: Contains utility functions for interacting with LLMs and retrieving search results.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sadhiin/perplexity-deep-search.git
   ```
2. Navigate to the project directory:
    ```bash
    cd perplexity-deep-search
    ```
3. Install dependencies
    ```bash
    uv sync
    ```
## Running the Application
To run the Streamlit app, execute the following command in the terminal:

    ```bash
    streamlit run main.py
    ```
## Configuration
Environment Variables: Set your API keys and other configurations in the `.env `file.
Python Version: Ensure Python 3.13 or higher is installed.
## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.