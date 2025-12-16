"""
Conversation manager for handling chat history and context retention.
"""

from typing import List, Dict, Optional
from datetime import datetime


class ConversationManager:
    """Manages conversation history and context for the RAG system."""
    
    def __init__(self):
        """Initialize the conversation manager."""
        self.messages: List[Dict] = []
        self.context: Dict = {}  # Store extracted context (wells, formations, etc.)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (sources, tools used, etc.)
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
    
    def get_conversation_history(self, max_messages: int = 10) -> List[Dict]:
        """
        Get recent conversation history.
        
        Args:
            max_messages: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
    
    def extract_context_from_history(self) -> Dict:
        """
        Extract context from conversation history (wells, formations, curves mentioned).
        
        Returns:
            Dictionary with extracted context
        """
        context = {
            'wells': set(),
            'formations': set(),
            'curves': set(),
            'last_query_type': None,
            'last_formation': None,
            'last_well': None
        }
        
        # Look at recent messages
        recent_messages = self.get_conversation_history(5)
        
        for msg in recent_messages:
            content = msg.get('content', '').lower()
            metadata = msg.get('metadata', {})
            
            # Extract from metadata if available
            if 'well_name' in metadata:
                context['wells'].add(metadata['well_name'])
            if 'formation' in metadata:
                context['formations'].add(metadata['formation'])
            if 'curve' in metadata:
                context['curves'].add(metadata['curve'])
            
            # Extract from content
            # Well names pattern: "15/9-F-1" or "NO_15/9-19_A"
            import re
            well_pattern = r'\b(\d+[/-]\d+[-/][A-Z0-9\s]+)\b'
            wells = re.findall(well_pattern, content, re.IGNORECASE)
            for well in wells:
                context['wells'].add(well.strip())
            
            # Formation names
            formations = ['hugin', 'sleipner', 'skagerrak', 'smith bank', 'heather', 
                         'draupne', 'hod', 'ekofisk', 'ty', 'utsira']
            for formation in formations:
                if formation in content:
                    context['formations'].add(formation.title())
            
            # Curves
            curves = ['gr', 'phif', 'klogh', 'sw', 'vsh', 'bvw', 'porosity', 'permeability']
            for curve in curves:
                if curve in content:
                    context['curves'].add(curve.upper())
        
        # Get last query info
        if recent_messages:
            last_msg = recent_messages[-1]
            if last_msg.get('role') == 'user':
                analysis = last_msg.get('metadata', {}).get('analysis', {})
                if analysis:
                    context['last_query_type'] = analysis.get('query_type')
                    context['last_formation'] = analysis.get('formation')
                    context['last_well'] = analysis.get('well_name')
        
        # Convert sets to lists
        context['wells'] = list(context['wells'])
        context['formations'] = list(context['formations'])
        context['curves'] = list(context['curves'])
        
        return context
    
    def resolve_reference(self, query: str) -> Dict:
        """
        Resolve references in follow-up questions (e.g., "that formation", "the well").
        
        Args:
            query: Current query
            
        Returns:
            Dictionary with resolved references
        """
        query_lower = query.lower()
        context = self.extract_context_from_history()
        resolved = {}
        
        # Resolve "that formation", "the formation"
        if ('that formation' in query_lower or 'the formation' in query_lower or 
            'it' in query_lower) and context.get('last_formation'):
            resolved['formation'] = context['last_formation']
        
        # Resolve "that well", "the well"
        if ('that well' in query_lower or 'the well' in query_lower) and context.get('last_well'):
            resolved['well'] = context['last_well']
        
        # Resolve "it" based on last query type
        if 'it' in query_lower:
            if context.get('last_formation'):
                resolved['formation'] = context['last_formation']
            if context.get('last_well'):
                resolved['well'] = context['last_well']
        
        return resolved
    
    def clear_conversation(self) -> None:
        """Clear conversation history and context."""
        self.messages = []
        self.context = {}
    
    def get_conversation_summary(self) -> str:
        """
        Get a text summary of the conversation for LLM context.
        
        Returns:
            Conversation summary string
        """
        if not self.messages:
            return ""
        
        summary_parts = ["Recent conversation:"]
        for msg in self.get_conversation_history(5):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]  # Truncate
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)


if __name__ == "__main__":
    # Test the conversation manager
    manager = ConversationManager()
    
    manager.add_message('user', 'What is the shale volume in Hugin formation?', 
                       {'analysis': {'formation': 'Hugin', 'curve': 'VSH'}})
    manager.add_message('assistant', 'The shale volume in Hugin formation is...')
    
    context = manager.extract_context_from_history()
    print("Extracted context:", context)
    
    resolved = manager.resolve_reference("What about that formation?")
    print("Resolved reference:", resolved)

