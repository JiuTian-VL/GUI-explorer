# INPUTS: app_name, package_name, activity_list
TASK_GOAL_GENERATOR = """Given the screenshot of {app_name} and its available activities, generate a comprehensive list of practical user tasks that:

1. Start from the current screen shown in the screenshot
2. Can be completed within 10-30 steps
3. Utilize the app's full feature set based on the activity list
4. Are concrete and specific (like searching for a particular item rather than just "search")
5. Cover different user interaction patterns (viewing, editing, sharing, etc.)
6. Include both basic and advanced features
7. Represent realistic user behaviors and goals
8. Avoid excessive steps on form-filling or scrolling pages

Important context:
- App name: {app_name}
- Package name: {package_name} 
- Available activities (app screens/features):
```{activity_list}```

Format requirements:
1. List only the tasks without explanations or commentary
2. Each task should be a single, clear directive
3. Use specific examples (e.g., concrete search terms, actions, settings)
4. Include the expected outcome where relevant
5. Tasks should follow this pattern: [Starting action] + [Specific steps] + [End goal]

Example tasks from other apps (for reference only):
1. Search for "ocean waves" white noise, then sort results by most played
2. Open the first recommended video, then post "Great content!" as a comment
3. Play the trending video, then add it to your "Watch Later" playlist
4. Navigate to the comments section of a featured video, then like the top comment

Generate diverse tasks that would help a user explore and utilize all major features visible in the screenshot and implied by the activity list."""


# INPUTS: task_description, numeric_tag_of_element, ui_element_attributes, action
KNOWLEDGE_EXTRACTOR = """Objective: Describe the functionality of a specific UI element in a mobile app screenshot.

Input:
- Two screenshots: Before and after interacting with a UI element
- UI element marked with a numeric tag in the top-left corner
- Element number: {numeric_tag_of_element}
- Broader task context: {task_description}
- Action taken: {action}
- UI Element Attributes: 
  ```
  {ui_element_attributes}
  ```

Requirements for Functionality Description:
1. Concise: 1-2 sentences
2. Focus on general function, not specific details
3. Avoid mentioning the numeric tag
4. Use generic terms like "UI element" or appropriate pronouns

Example:
- Incorrect: "Tapping the element #3 displays David's saved recipes in the results panel"
- Correct: "Tapping this element will initiates a search and displays matching results"

Guidance:
- Describe the core action and immediate result of interacting with the UI element
- Prioritize clarity and generality in the description"""


# INPUTS: task_goal, knowledge_a, knowledge_b
RANKER = """Given the user instruction: {task_goal}, determine which of the following two knowledge entries is more useful.
Respond ONLY with a integer value:
1 means Knowledge A is strictly better.
2 means Knowledge B is strictly better.

Knowledge A: {knowledge_a}
Knowledge B: {knowledge_b}

Please provide your response:
"""


# INPUTS: task_goal, history, ui_elements, knowledge
REASONING = """## Role Definition
You are an Android operation AI that fulfills user requests through precise screen interactions.
The current screenshot and the same screenshot with bounding boxes and labels added are also given to you.

## Action Catalog
Available actions (STRICT JSON FORMAT REQUIRED):
1. Status Operations:
   - Task Complete: {{"action_type": "status", "goal_status": "complete"}}
   - Task Infeasible: {{"action_type": "status", "goal_status": "infeasible"}}
2. Information Actions:
   - Answer Question: {{"action_type": "answer", "text": "<answer_text>"}}
3. Screen Interactions:
   - Tap Element: {{"action_type": "click", "index": <visible_index>}}
   - Long Press: {{"action_type": "long_press", "index": <visible_index>}}
   - Scroll: Scroll the screen or a specific scrollable UI element. Use the `index` of the target element if scrolling a specific element, or omit `index` to scroll the whole screen. {{"action_type": "scroll", "direction": <"up"|"down"|"left"|"right">, "index": <optional_target_index>}}
4. Input Operations:
   - Text Entry: {{"action_type": "input_text", "text": "<content>", "index": <text_field_index>}}
   - Keyboard Enter: {{"action_type": "keyboard_enter"}}
5. Navigation:
   - Home Screen: {{"action_type": "navigate_home"}}
   - Back Navigation: {{"action_type": "navigate_back"}}
6. System Actions:
   - Launch App: {{"action_type": "open_app", "app_name": "<exact_name>"}}
   - Wait Refresh: {{"action_type": "wait"}}

## Current Objective
User Goal: {task_goal}

## Execution Context
Action History:
{history}

Visible UI Elements (Only interact with *visible=true elements):
{ui_elements}

## Core Strategy
1. Path Optimization:
   - Prefer direct methods (e.g., open_app > app drawer navigation)
   - Always use the `input_text` action for entering text into designated text fields.
   - Verify element visibility (`visible=true`) before attempting any interaction (click, long_press, input_text). Do not interact with elements marked `visible=false`.
   - Use `scroll` when necessary to bring off-screen elements into view. Prioritize scrolling specific containers (`index` provided) over full-screen scrolls if possible.

2. Error Handling Protocol:
   - Switch approach after ≥ 2 failed attempts
   - Prioritize scrolling (`scroll` action) over force-acting on invisible elements
   - If an element is not visible, use `scroll` in the likely direction (e.g., 'down' to find elements below the current view).
   - Try opposite scroll direction if initial fails (up/down, left/right)
   - If the `open_app` action fails to correctly open the app, find the corresponding app in the app drawer and open it.

3. Information Tasks:
   - MANDATORY: Use answer action for questions
   - Verify data freshness (e.g., check calendar date)

## Expert Techniques
Here are some tips for you:
{knowledge}

## Response Format
STRICTLY follow:
Reasoning: [Step-by-step analysis covering:
           - Visibility verification
           - History effectiveness evaluation
           - Alternative approach comparison
           - Consideration of scrolling if needed]
Action: [SINGLE JSON action from catalog]

Generate response:
"""

# INPUTS: task_goal, before_ui_elements, after_ui_elements, action, reasoning
SUMMARY="""
Goal: {task_goal}

Before screenshot elements:
{before_ui_elements}

After screenshot elements:
{after_ui_elements}

Action: {action}
Reasoning: {reasoning}

Provide a concise single-line summary (under 50 words) of this step by comparing screenshots and action outcome. Include:
- What was intended
- Whether it succeeded
- Key information for future actions
- Critical analysis if action/reasoning was flawed
- Important data to remember across apps

For actions like 'answer' or 'wait' with no screen change, assume they worked as intended.

Summary:
"""