import asyncio
import os
import urllib3
from pydantic import BaseModel, Field

#disable the urllib3 warning about SSL, handles http req
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner
from agents.tool import FileSearchTool # tool that helps search for files
openai_api_key = os.getenv("OPENAI_API_KEY")
from stripe_agent_toolkit.openai.toolkit import StripeAgentToolkit # this imports a Stripe toolkit that can interact with the Stripe API (used for payment processing)

# Update Stripe configuration to explicitly set currency to USD
stripe_agent_toolkit = StripeAgentToolkit(
    secret_key='your_stripe_api_key',
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
                    "currency": "usd"  # Explicitly set currency to USD
                }
            },
            "invoice_items": {
                "create": True,
                "default_params": {
                    "currency": "usd"  # Explicitly set currency to USD
                }
            },
            "invoices": {
                "create": True,
                "update": True,
                "default_params": {
                    "currency": "usd"  # Explicitly set currency to USD
                }
            }
        }
    },
)
#template for invoice info
class InvoiceOutput(BaseModel):
    name: str = Field(description="The name of the customer")
    email: str = Field(description="The email of the customer")
    service: str = Field(description="The service that the customer is invoiced for")
    amount_due: int = Field(description="The dollar amount due for the invoice. Convert text to dollar amounts if needed.")

class InvoiceListOutput(BaseModel):
    invoices: list[InvoiceOutput]

file_search_agent = Agent(
    name="File Search Agent",
    instructions="You are an expert at searching for financial documents.",
    tools=[
        FileSearchTool(
            max_num_results=50,
            vector_store_ids=['vs_67d202bbb540819198de06c08eb95586'],
        )
    ],
    output_type=InvoiceListOutput,
)

# Update the invoice agent instructions with a more detailed workflow
invoice_agent = Agent(
    name="Invoice Agent",
    instructions="""You are an expert at using the Stripe API to create, finalize, and send invoices to customers. Always use USD as the currency for all operations.

Follow this exact sequence when creating invoices:
1. Create a customer with the provided name and email
2. Create a product with the service description
3. Create a price for that product with the specified amount in USD (convert to cents)
4. Create an invoice for the customer (this will return an invoice ID)
5. Use the invoice ID from step 4 to add invoice items to the invoice
6. Finalize the invoice using the invoice ID
7. Send the invoice using the invoice ID

IMPORTANT: Always create the invoice BEFORE creating invoice items, and always use the actual invoice ID returned from the invoice creation step.
""",
    tools=stripe_agent_toolkit.get_tools(),
)

# KL to chat, KK to generate
async def main():
    assignment = "Search for all customers that haven't paid across all of my documents. For each, create, finalize, and send an invoice."
    
    outstanding_invoices = await Runner.run(
        file_search_agent,
        assignment,
    )
    
    invoices_to_send = outstanding_invoices.final_output.invoices
    
    for invoice in invoices_to_send:
        print(invoice.name, invoice.email, invoice.service, invoice.amount_due)
        
    # Iterate through each invoice and create a task
    for invoice in invoices_to_send:
        print(f"Initiating invoice generation for {invoice.name} ({invoice.email}) for {invoice.service} for ${invoice.amount_due}.")
        
        invoice_task = await Runner.run(
            invoice_agent,
            f"Create, finalize, and send an invoice to {invoice.name} ({invoice.email}) for {invoice.service} for ${invoice.amount_due}."
        )
        print(invoice_task.final_output)

if __name__ == "__main__":
    asyncio.run(main())