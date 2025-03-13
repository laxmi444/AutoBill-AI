import streamlit as st
import asyncio
import os
import urllib3
from pydantic import BaseModel, Field
import pandas as pd
import nest_asyncio
from typing import List

# Apply nest_asyncio to allow running asyncio in streamlit
nest_asyncio.apply()

# Disable the urllib3 warning about SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner
from agents.tool import FileSearchTool

from stripe_agent_toolkit.openai.toolkit import StripeAgentToolkit

# Initialize session state for storing results
if 'outstanding_invoices' not in st.session_state:
    st.session_state.outstanding_invoices = None
if 'invoice_results' not in st.session_state:
    st.session_state.invoice_results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'step' not in st.session_state:
    st.session_state.step = 1

# Pydantic models
class InvoiceOutput(BaseModel):
    name: str = Field(description="The name of the customer")
    email: str = Field(description="The email of the customer")
    service: str = Field(description="The service that the customer is invoiced for")
    amount_due: int = Field(description="The dollar amount due for the invoice. Convert text to dollar amounts if needed.")

class InvoiceListOutput(BaseModel):
    invoices: List[InvoiceOutput]

# Page configuration
st.set_page_config(page_title="Stripe Invoice Generator", layout="wide")
st.title("Stripe Invoice Generator")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Stripe API Key input (masked)
    stripe_api_key = st.text_input(
        "Stripe API Key", 
        value="stripe_api_key",
        type="password"
    )
    
    # Vector store ID input
    vector_store_id = st.text_input(
        "Vector Store ID", 
        value="vs_67d202bbb540819198de06c08eb95586"
    )
    
    # Currency selection
    currency = st.selectbox(
        "Currency",
        options=["usd", "eur", "gbp", "jpy"],
        index=0
    )
    
    st.divider()
    
    # Run button
    if st.button("Search & Process Unpaid Invoices", type="primary"):
        st.session_state.processing = True
        st.session_state.step = 1
        st.session_state.invoice_results = []

# Initialize Stripe Agent Toolkit with configuration
def init_stripe_toolkit(api_key, currency):
    return StripeAgentToolkit(
        secret_key=api_key,
        configuration={
            "actions": {
                "customers": {
                    "create": True,
                },
                "products": {
                    "create": True,
                },
                "prices": {
                    "create": True,
                    "default_params": {
                        "currency": currency
                    }
                },
                "invoice_items": {
                    "create": True,
                    "default_params": {
                        "currency": currency
                    }
                },
                "invoices": {
                    "create": True,
                    "update": True,
                    "default_params": {
                        "currency": currency
                    }
                }
            }
        },
    )

# Initialize agents
def init_agents(stripe_api_key, vector_store_id, currency):
    stripe_agent_toolkit = init_stripe_toolkit(stripe_api_key, currency)
    
    file_search_agent = Agent(
        name="File Search Agent",
        instructions="You are an expert at searching for financial documents.",
        tools=[
            FileSearchTool(
                max_num_results=50,
                vector_store_ids=[vector_store_id],
            )
        ],
        output_type=InvoiceListOutput,
    )

    invoice_agent = Agent(
        name="Invoice Agent",
        instructions=f"""You are an expert at using the Stripe API to create, finalize, and send invoices to customers. Always use {currency.upper()} as the currency for all operations.

Follow this exact sequence when creating invoices:
1. Create a customer with the provided name and email
2. Create a product with the service description
3. Create a price for that product with the specified amount in {currency.upper()} (convert to cents)
4. Create an invoice for the customer (this will return an invoice ID)
5. Use the invoice ID from step 4 to add invoice items to the invoice
6. Finalize the invoice using the invoice ID
7. Send the invoice using the invoice ID

IMPORTANT: Always create the invoice BEFORE creating invoice items, and always use the actual invoice ID returned from the invoice creation step.
""",
        tools=stripe_agent_toolkit.get_tools(),
    )
    
    return file_search_agent, invoice_agent

# Async function to search for unpaid invoices
async def search_unpaid_invoices(file_search_agent):
    assignment = "Search for all customers that haven't paid across all of my documents. For each, create, finalize, and send an invoice."
    
    with st.spinner("Searching for unpaid invoices..."):
        outstanding_invoices = await Runner.run(
            file_search_agent,
            assignment,
        )
    
    return outstanding_invoices.final_output

# Async function to process and send an invoice
async def process_invoice(invoice_agent, invoice):
    prompt = f"Create, finalize, and send an invoice to {invoice.name} ({invoice.email}) for {invoice.service} for ${invoice.amount_due}."
    
    # Create a placeholder for URL list
    url_container = st.empty()
    
    # Display basic invoice processing information
    st.write(f"Processing invoice for {invoice.name}...")
    
    # Standard Stripe API URLs for invoice flow
    stripe_urls = [
        "https://api.stripe.com/v1/customers", 
        "https://api.stripe.com/v1/products",
        "https://api.stripe.com/v1/prices",
        "https://api.stripe.com/v1/invoices",
        "https://api.stripe.com/v1/invoiceitems",
        f"https://api.stripe.com/v1/invoices/[invoice_id]/finalize",
        f"https://api.stripe.com/v1/invoices/[invoice_id]/send"
    ]
    
    # Display URLs as a simple list
    # with url_container.container():
    #     st.write("**Stripe API Endpoints:**")
    #     for url in stripe_urls:
    #         st.write(f"- {url}")
    
    # Process the invoice
    with st.spinner(f"Processing invoice for {invoice.name}..."):
        invoice_task = await Runner.run(
            invoice_agent,
            prompt
        )
        
        # Add debug output to see what's returned
        st.write("**Debug - Agent Output:**")
        st.code(str(invoice_task.final_output))
        print(invoice_task.final_output)  # Also print to console/logs
    
    # Return the result with URLs
    return {
        "invoice": invoice,
        "result": invoice_task.final_output,
        "urls": stripe_urls
    }

# Function to run the invoice search and processing workflow
def run_invoice_workflow():
    if not st.session_state.processing:
        return
    
    # Step 1: Search for unpaid invoices
    if st.session_state.step == 1:
        st.subheader("Step 1: Searching for Unpaid Invoices")
        
        # Initialize agents with current configurations
        file_search_agent, invoice_agent = init_agents(
            stripe_api_key, 
            vector_store_id,
            currency
        )
        
        # Run the search asynchronously
        outstanding_invoices = asyncio.run(search_unpaid_invoices(file_search_agent))
        st.session_state.outstanding_invoices = outstanding_invoices
        
        # Move to next step
        st.session_state.step = 2
        st.rerun()
    
    # Step 2: Display found invoices and initiate processing
    elif st.session_state.step == 2:
        st.subheader("Step 2: Review Unpaid Invoices")
        
        invoices = st.session_state.outstanding_invoices.invoices
        
        if not invoices:
            st.success("No unpaid invoices found!")
            st.session_state.processing = False
            return
        
        # Display the found invoices in a table
        invoice_data = [{
            "Name": inv.name,
            "Email": inv.email,
            "Service": inv.service,
            "Amount Due": f"${inv.amount_due}"
        } for inv in invoices]
        
        st.dataframe(pd.DataFrame(invoice_data), use_container_width=True)
        
        # Continue button
        if st.button("Process All Invoices"):
            st.session_state.step = 3
            st.rerun()
    
    # Step 3: Process each invoice
    elif st.session_state.step == 3:
        st.subheader("Step 3: Processing Invoices")
        
        # Initialize agents with current configurations
        file_search_agent, invoice_agent = init_agents(
            stripe_api_key, 
            vector_store_id,
            currency
        )
        
        invoices = st.session_state.outstanding_invoices.invoices
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Process each invoice
        for i, invoice in enumerate(invoices):
            # Skip if already processed
            if i < len(st.session_state.invoice_results):
                continue
                
            st.write(f"Processing invoice {i+1} of {len(invoices)}: {invoice.name} for {invoice.service}")
            
            # Process this invoice
            result = asyncio.run(process_invoice(invoice_agent, invoice))
            st.session_state.invoice_results.append(result)
            
            # Update progress
            progress = (i + 1) / len(invoices)
            progress_bar.progress(progress)
            
            if i < len(invoices) - 1:
                st.rerun()
        
        # All done, move to results
        st.session_state.step = 4
        st.rerun()
    
    # Step 4: Show final results
    elif st.session_state.step == 4:
        st.subheader("Step 4: Invoice Processing Results")
        
        # Display results in an expandable format
        for i, result in enumerate(st.session_state.invoice_results):
            invoice = result["invoice"]
            with st.expander(f"Invoice {i+1}: {invoice.name} - ${invoice.amount_due} for {invoice.service}"):
                st.write(f"**Email:** {invoice.email}")
                st.write(f"**Service:** {invoice.service}")
                st.write(f"**Amount:** ${invoice.amount_due}")
                
                # Display final output as a list
                st.write("**Invoice Generation Results:**")
                if "result_list" in result:
                    for item in result["result_list"]:
                        if item.strip():  # Only show non-empty lines
                            st.write(f"- {item.strip()}")
                else:
                    # Fallback if result_list is not available
                    result_str = str(result["result"])
                    for line in result_str.split('\n'):
                        if line.strip():
                            st.write(f"- {line.strip()}")
        
        # Completion message
        st.success(f"Successfully processed {len(st.session_state.invoice_results)} invoices!")
        
        # Reset button
        if st.button("Start New Search"):
            st.session_state.processing = False
            st.session_state.step = 1
            st.session_state.outstanding_invoices = None
            st.session_state.invoice_results = []
            st.rerun()

# Run the workflow if processing flag is set
if st.session_state.processing:
    run_invoice_workflow()
else:
    # Show instructions when not processing
    st.info("""
    This application searches through your documents for customers with unpaid invoices, 
    then creates and sends Stripe invoices automatically.
    
    To begin:
    1. Configure your Stripe API key and other settings in the sidebar
    2. Click "Search & Process Unpaid Invoices" to start
    """)
    
    # Example output display for demonstration
    with st.expander("Example Output (Click to expand)"):
        st.write("This is an example of what the results will look like after processing:")
        
        example_df = pd.DataFrame([
            {"Name": "John Doe", "Email": "john@example.com", "Service": "Web Development", "Amount Due": "$500"},
            {"Name": "Jane Smith", "Email": "jane@example.com", "Service": "Logo Design", "Amount Due": "$250"},
        ])
        
        st.dataframe(example_df, use_container_width=True)

# Add additional information about workflow
with st.sidebar:
    st.markdown("### How It Works")
    st.markdown("""
    **Invoice Creation Workflow:**
    1. Search documents for unpaid invoices
    2. For each customer found:
       - Create customer in Stripe
       - Create product with service description
       - Create price for the product
       - Create invoice for customer
       - Add invoice items to the invoice
       - Finalize the invoice
       - Send the invoice
    """)
