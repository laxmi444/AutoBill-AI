import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
import openai  
from pydantic import BaseModel, Field

class MCPConfig(BaseModel):
    """Configuration for the Model Context Protocol client."""
    api_key: str = Field(..., description="API key for the OpenAI API")  
    model: str = Field("gpt-4", description="Model identifier to use")  
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for sampling")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    context_window_tokens: int = Field(128000, description="Maximum context window size in tokens")  #adjusted for gpt4

class MCP:
    """
    Model Context Protocol client for managing interactions with OpenAI models
    while maintaining context across multiple calls.
    """
    
    def __init__(self, config: MCPConfig):
        """
        Initialize the MCP client with the given configuration.
        
        Args:
            config: Configuration for the MCP client
        """
        self.config = config
        self.client = openai.AsyncClient(api_key=config.api_key)  # Changed to OpenAI client
        self.conversation_history: List[Dict[str, str]] = []
        self.active_context_tokens = 0
        self.call_count = 0
        self.system_message = None  #addef to support system messages
    
    def set_system_message(self, message: str):
        """
        Set a system message for the conversation.
        
        Args:
            message: The system message to set
        """
        self.system_message = message
    
    async def complete(self, prompt: str) -> str:
        """
        Complete the given prompt using the configured model.
        
        Args:
            prompt: The prompt to complete
            
        Returns:
            The completion text
        """
        # add the prompt to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        
        ##create messages format from conversation history, managing context as needed
        messages = self._prepare_messages()
        
        #call api
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            self.call_count += 1
            
            #extract the response text
            response_text = response.choices[0].message.content
            
            #add the response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            #iupdate token count
            self._update_token_count(prompt, response_text)
            
            return response_text

        except Exception as e:
            #handle any errors
            print(f"Error calling OpenAI API: {str(e)}")
            raise e
    
    def _prepare_messages(self) -> List[Dict[str, str]]:
        """
        Prepare messages for the API call, managing context as needed.
        
        Returns:
            List of message objects for the API call
        """
        messages = []
        
        ##add system message if present
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        
        # If we are within context window limits, use the full history
        if self.active_context_tokens <= self.config.context_window_tokens * 0.9:
            messages.extend(self.conversation_history)
            return messages
        
        # Otherwise, we need to trim the history
        # Start with the most recent user message
        recent_messages = [self.conversation_history[-1]]
        
        # Add previous messages until we hit context limits
        for message in reversed(self.conversation_history[:-1]):
            if len(recent_messages) >= 10:  # Arbitrary limit for demonstration
                break
            recent_messages.insert(0, message)
        
        messages.extend(recent_messages)
        return messages
    
    def _update_token_count(self, prompt: str, response: str):
        """
        Update the active context token count.
        This is a simplified implementation - in production, use a proper tokenizer.
        
        Args:
            prompt: The prompt that was sent
            response: The response that was received
        """
        # Very rough estimation - 4 chars ≈ 1 token
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        
        # update active context tokens
        self.active_context_tokens += prompt_tokens + response_tokens
    
    def reset_conversation(self):
        """Reset the conversation history and token count."""
        self.conversation_history = []
        self.active_context_tokens = 0
        self.system_message = None  #also reset system message
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the MCP client's usage.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            "active_context_tokens": self.active_context_tokens,
            "call_count": self.call_count,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "conversation_turns": len(self.conversation_history) // 2,
            "has_system_message": self.system_message is not None
        }

#example
async def example():
    config = MCPConfig(
        api_key="your_openai_api_key",
        model="gpt-4",
        max_tokens=1024
    )
    
    mcp = MCP(config)
    #set a system message(optional)
    mcp.set_system_message("You are a helpful AI assistant.")
    response = await mcp.complete("Hello, can you help me with something?")
    print(response)

if __name__ == "__main__":
    asyncio.run(example())