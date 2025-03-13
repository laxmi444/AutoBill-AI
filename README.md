# Stripe Invoice Generator

An AI-powered tool for automatically finding unpaid invoices in your documents and generating invoices in Stripe. The tool is available in two versions:

1. **Command-line script** - For automation and integration with other systems
2. **Streamlit web app** - For a user-friendly interface with visual feedback

## Overview

This tool uses advanced AI agents to:

1. Search through your documents for customers who haven't paid invoices
2. Create Stripe customers, products, and invoices
3. Finalize and send invoices automatically

## Requirements

### Common Dependencies

- Python 3.8+
- `agents` library (for AI agent functionality)
- `stripe_agent_toolkit` (for Stripe API integration)
- `pydantic` (for data validation)
- `python-dotenv` (for environment variables)
- `urllib3` (for HTTP requests)

### Streamlit Version Additional Dependencies

- `streamlit` (for web interface)
- `pandas` (for data handling)
- `nest_asyncio` (for running async code in Streamlit)

## Installation

1. Clone the repository or download the script files
2. Install the required dependencies:

```bash
# For command-line version
pip install pydantic python-dotenv urllib3 openai-agents stripe_agent_toolkit

# For Streamlit version (includes all dependencies)
pip install pydantic python-dotenv urllib3 openai-agents stripe_agent_toolkit streamlit pandas nest_asyncio
```

3. Set up your environment variables by creating a `.env` file with your API keys:

```
STRIPE_API_KEY=your_stripe_api_key
VECTOR_STORE_ID=your_vector_store_id
```

## Command-Line Version Usage

The command-line script (`app.py`) is designed for automation and can be run from the terminal:

```bash
python app.py
```

### Key Features

- Automatically searches all documents for unpaid invoices
- Creates Stripe invoices following a precise 7-step workflow
- Outputs progress and results to the console
- Suitable for cron jobs, CI/CD pipelines, or server-side automation

### Code Explanation

The script performs these operations:

1. Initializes the Stripe toolkit with your API key
2. Sets up two AI agents:
   - `file_search_agent` - Searches documents for unpaid invoices
   - `invoice_agent` - Handles Stripe API operations
3. Runs the search agent to find customers with unpaid invoices
4. For each customer found, runs the invoice agent to create and send an invoice
5. Prints results to the console

## Streamlit App Version Usage

The Streamlit app (`app_streamlit.py`) provides a user-friendly web interface:

```bash
streamlit run app_streamlit.py
```

### Key Features

- Interactive web interface with configuration options
- Step-by-step guided workflow
- Visual display of found invoices with detailed information
- Progress tracking for invoice processing
- Expandable results with detailed API responses
- Currency selection and configuration options

### App Workflow

1. **Configuration** - Set up your Stripe API key and options in the sidebar
2. **Search** - Click "Search & Process Unpaid Invoices" to start
3. **Review** - See a table of found unpaid invoices
4. **Process** - Click to process and send all invoices
5. **Results** - View detailed results for each processed invoice

## Configuration Options

Both versions support these configuration options:

### Stripe API Configuration

- **API Key**: Your Stripe secret key
- **Currency**: Default is USD, but can be changed in the Streamlit version
- **Vector Store ID**: ID for the document search functionality

### Invoice Agent Instructions

The invoice agent follows this precise workflow:

1. Create a customer with the provided name and email
2. Create a product with the service description
3. Create a price for that product with the specified amount
4. Create an invoice for the customer (this will return an invoice ID)
5. Use the invoice ID from step 4 to add invoice items to the invoice
6. Finalize the invoice using the invoice ID
7. Send the invoice using the invoice ID

## Version Comparison

| Feature           | Command-Line Version    | Streamlit App Version                           |
| ----------------- | ----------------------- | ----------------------------------------------- |
| User Interface    | Terminal output         | Web-based UI with interactive elements          |
| Configuration     | Via code or `.env` file | Via UI controls in sidebar                      |
| Progress Tracking | Text output to console  | Visual progress bars and step indicators        |
| Results Display   | Text output to console  | Interactive expandable sections with formatting |
| Currency Options  | Hardcoded (USD)         | Selectable from dropdown                        |
| Deployment        | Can run headless        | Requires web server or local access             |
| Automation        | Suitable for automation | Designed for interactive use                    |

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors**

   - The script disables SSL warnings, but you may need to install proper certificates in your environment

2. **API Key Issues**

   - Ensure your Stripe API key is valid and has the necessary permissions
   - Check that the key is properly configured in the `.env` file or UI

3. **No Invoices Found**

   - Verify your vector store ID is correct
   - Ensure your documents contain unpaid invoice information

4. **Streamlit Display Issues**

   - Try running `streamlit clear` to clear the cache
   - Update Streamlit with `pip install --upgrade streamlit`

5. **Async Errors in Streamlit**
   - The app uses `nest_asyncio` to handle async code in Streamlit
   - If you encounter issues, try restarting the Streamlit server

## Security Considerations

- The Stripe API key is sensitive information. In the command-line version, store it in the `.env` file and ensure this file is not committed to version control.
- In the Streamlit app, the API key is masked in the UI but still stored in session state. Deploy the app securely.
- Consider implementing access controls if deploying the Streamlit app in a multi-user environment.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
