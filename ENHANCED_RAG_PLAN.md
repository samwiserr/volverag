# Enhanced RAG System with Gemini LLM, Tools, and Formation Visualization

## Overview

Enhance the RAG system with:
1. **Gemini LLM Integration** with strict data grounding
2. **Computational Tools** for on-the-fly analysis and plotting
3. **Formation Visualization** with log curves and formation highlighting
4. **Conversational Frontend** with chat history and context retention

## Key Requirements

### Strict Data Grounding
- **ALL answers must come from retrieved data only**
- No external knowledge or hallucinations
- Validate all claims against data sources
- Clear indication when data is insufficient

### Formation Visualization
- When formation questions are asked, offer option to display:
  - Well log with specified curves (e.g., GR for shale volume)
  - Formation interval highlighted/shaded on the log
  - Allow users to verify results by reading curve values
- Example: "What is the shale volume in Hugin formation?" → Option to show GR curve with Hugin formation highlighted

## Implementation Plan

### 1. Computational Tools Framework

**File**: `src/tools.py` (new)

Create a tool system for on-demand computations:

**Tools to Implement**:
1. **`plot_formation_log(well_name, formation_name, curves)`**
   - Load LAS data for well
   - Extract formation depth interval from formation tops
   - Plot specified curves (e.g., GR, PHIF, VSH)
   - Highlight formation interval with shading/color
   - Return interactive Altair chart

2. **`get_formation_interval(well_name, formation_name)`**
   - Get formation top and base depths
   - Return depth range for formation

3. **`plot_log_curves(well_name, curves, depth_range)`**
   - Plot multiple curves for specified depth range
   - Support custom depth ranges

4. **`calculate_formation_statistics(well_name, formation_name, curve)`**
   - Calculate statistics for curve within formation interval
   - Return mean, min, max, std

5. **`compare_formations(well_name, formations, curve)`**
   - Compare same curve across multiple formations in one well
   - Visual comparison

**Tool Schema Format**:
```python
TOOLS = [
    {
        "name": "plot_formation_log",
        "description": "Plot well log curves with formation interval highlighted",
        "parameters": {
            "well_name": "str - Name of the well",
            "formation_name": "str - Name of the formation",
            "curves": "List[str] - Curves to plot (e.g., ['GR', 'PHIF', 'VSH'])"
        }
    },
    ...
]
```

### 2. Enhanced Visualization Module

**File**: `src/visualization.py` (enhance)

**New Methods**:
- `plot_formation_with_highlight(well_name, formation_name, curves, log_data, formation_tops)`
  - Create log plot with formation interval highlighted
  - Use Altair's `mark_rect` for formation shading
  - Overlay curves on top
  - Add formation labels

- `get_relevant_curves_for_query(query, curve_mentioned)`
  - Map query to relevant curves:
    - Shale volume → GR (Gamma Ray)
    - Porosity → PHIF, PORD
    - Permeability → KLOGH
    - Water saturation → SW
  - Return suggested curves for visualization

### 3. Gemini LLM Integration with Function Calling

**File**: `src/answer_generator.py` (enhance)

**Key Features**:
- Support Gemini via `langchain-google-genai`
- Use Gemini's function calling capability
- Strict grounding: System prompt enforces data-only answers
- Tool selection: LLM decides which tools to use based on query
- Tool execution: Execute selected tools and incorporate results

**System Prompt**:
```
You are a helpful assistant for oil and gas well data analysis. 
CRITICAL: You can ONLY answer using:
1. The provided well data context
2. Results from computational tools (plotting, statistics)
3. Retrieved formation tops data

DO NOT use any external knowledge. If information is not in the provided data, 
state that the data is not available rather than guessing.

When questions involve formations, you should offer to display the formation 
on a well log with relevant curves highlighted.
```

**Function Calling Flow**:
1. User query → Analyze intent
2. LLM selects appropriate tools
3. Execute tools with parameters
4. LLM generates answer using tool results + retrieved context
5. Return answer with visualization option

### 4. Data Access Layer

**File**: `src/data_access.py` (new)

- Cache LAS file data for quick access
- Load LAS data on-demand for plotting
- Map well names to LAS file paths
- Extract formation intervals from formation tops
- Filter log data by depth range

**Key Methods**:
- `get_well_log_data(well_name)` - Load LAS data for well
- `get_formation_depth_range(well_name, formation_name)` - Get formation MD range
- `filter_log_by_depth(log_data, start_depth, end_depth)` - Filter data

### 5. Conversational Frontend

**File**: `src/frontend.py` (transform)

**New Features**:
- **Chat Interface**: Message-based UI
- **Message History**: Store in session state
- **Visualization Display**: Show plots inline in chat
- **Formation Visualization Button**: "Show Formation on Log" button when relevant
- **Context Retention**: Pass conversation history to LLM
- **Tool Usage Display**: Show which tools were used

**UI Structure**:
```
┌─────────────────────────────────────┐
│  Chat Messages (scrollable)        │
│  ┌─────────────┐                   │
│  │ User: What is shale volume...   │
│  └─────────────┘                   │
│  ┌─────────────────────────────┐   │
│  │ Assistant: Answer...         │   │
│  │ [Show Formation on Log] [📊] │   │
│  └─────────────────────────────┘   │
│  ┌─────────────┐                   │
│  │ [Chart displayed here]          │
│  └─────────────┘                   │
├─────────────────────────────────────┤
│  [Input] [Send] [Clear Chat]        │
└─────────────────────────────────────┘
```

### 6. Conversation Manager

**File**: `src/conversation_manager.py` (new)

- Manage chat history
- Extract context from previous messages
- Handle follow-up questions
- Resolve references ("that formation", "the well")
- Track which tools were used

### 7. Enhanced Query Processing

**File**: `src/rag_query.py` (enhance)

- Accept conversation context
- Pass context to answer generator
- Return tool execution results
- Include visualization options in response

## Implementation Details

### Formation Visualization Example

**Query**: "What is the source of shale volume in Hugin formation in well 15/9-F-1?"

**Response Flow**:
1. Retrieve Hugin formation data for well 15/9-F-1
2. Calculate shale volume statistics
3. Generate answer mentioning GR curve as source
4. Offer: "Would you like to see the Gamma Ray curve with Hugin formation highlighted?"
5. If user clicks → Execute `plot_formation_log("15/9-F-1", "Hugin", ["GR", "VSH"])`
6. Display chart with:
   - GR curve plotted
   - Hugin formation interval shaded/highlighted
   - Depth scale
   - Formation labels

### Curve Selection Logic

**Query → Relevant Curves Mapping**:
- "shale volume" / "VSH" → GR (Gamma Ray), VSH
- "porosity" / "PHIF" → PHIF, PORD, NPHI
- "permeability" / "KLOGH" → KLOGH, PHIF (for crossplot)
- "water saturation" / "SW" → SW, RW, PHIF
- "formation" (general) → GR, PHIF, VSH (standard suite)

### Strict Grounding Implementation

1. **System Prompt**: Explicitly forbid external knowledge
2. **Validation**: Check all numerical claims against retrieved data
3. **Source Attribution**: Every answer must cite data sources
4. **Uncertainty Handling**: When data insufficient, state clearly
5. **Tool Results Only**: LLM can only use tool outputs, not external knowledge

## Files to Create/Modify

### New Files
1. `src/tools.py` - Tool definitions and execution
2. `src/data_access.py` - On-demand LAS data loading
3. `src/conversation_manager.py` - Chat history management

### Modified Files
1. `src/answer_generator.py` - Add Gemini + function calling
2. `src/visualization.py` - Add formation highlighting
3. `src/rag_query.py` - Support tools and conversation context
4. `src/frontend.py` - Transform to chat interface
5. `requirements.txt` - Add Gemini dependencies
6. `.env.example` - Add GEMINI_API_KEY

## Dependencies

```txt
langchain-google-genai>=1.0.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
```

## Testing Scenarios

1. **Formation Query with Visualization**:
   - "What is the shale volume in Hugin formation?"
   - Verify answer is data-grounded
   - Test "Show Formation on Log" button
   - Verify GR curve displayed with formation highlighted

2. **Follow-up Questions**:
   - "What about the Sleipner formation?"
   - Verify context retention
   - Verify formation visualization for new formation

3. **Tool Execution**:
   - Test each tool independently
   - Verify tool results are accurate
   - Test error handling

4. **Strict Grounding**:
   - Ask question requiring external knowledge
   - Verify system says "data not available"
   - Verify no hallucinations

## Benefits

1. **Visual Verification**: Users can verify answers by reading curve values
2. **Better Understanding**: Visual context helps interpret results
3. **Natural Interaction**: Conversational interface feels more intuitive
4. **Trust**: Strict grounding ensures reliable answers
5. **Flexibility**: Tools enable on-demand analysis

