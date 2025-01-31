import base64
from logging import raiseExceptions

import requests
import io
import os
from anthropic import Anthropic
import google.generativeai as genai
from abc import ABC, abstractmethod
import re
import logging
from PIL import Image, ImageDraw, ImageFont
import experiment_utilities as util

# open source deps
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DEBUG = False

class Agent(ABC):
    """
    Abstract base class for agents.
    Provides a template for specific agents with standardized methods for interaction.
    """

    @abstractmethod
    def act(self, observation, i):
        """
        Abstract method to perform an action in the environment.

        Parameters:
            action (any): The action to perform.

        Returns:
            None
        """
        pass


class AIAgent(Agent):
    def __init__(self, episode_logger, move_sequence):
        self.episode_logger = episode_logger
        self.move_sequence = move_sequence
        #self.move_sequence.pop(-1)
        self.delay = 0

    def act(self, observation, i):
        if not self.move_sequence:
            logging.warning("AIAgent path empty")
            return "path empty"
        else:
            action = self.move_sequence.pop(0)
            self.episode_logger.info(f"action {action}")
            return action


class UserAgent(Agent):
    """
    A simple interactive agent that simulates the role of an LLM for user interaction.

    Instead of generating responses automatically, it prompts the user for manual input
    and returns that input as the action.
    """

    def __init__(self):
        pass

    def act(self, observation):
        """
        Simulates the process of responding to an observation by asking for user input.

        Args:
            observation (str): A placeholder for the environment's observation. It is not used
                               for decision-making in this class, but passed for compatibility
                               with LLM-style interfaces.

        Returns:
            str: The user's input, simulating the action.
        """
        # Get user input (no validation)
        action = input("Enter action: ").strip()

        return action


class LLMAgent:
    """Parent class for LLM-based agents"""

    def __init__(self, episode_path, episode_logger, api_key_file_path, instruction_prompt_file_path,
                 visual_state_embedding='label', single_images=True, COT=False, delay=0, max_history=0):
        self.max_history = max_history
        self.api_keys = load_api_keys(api_key_file_path)
        self.goal_state = None
        self.single_images = single_images
        self.COT = COT
        self.color_codes = util.load_params_from_json('color_codes.json')
        self.visual_state_embedding = visual_state_embedding
        self.episode_logger = episode_logger
        self.delay = delay

        # Create the 'obs' subdirectory inside the save path
        self.obs_dir = os.path.join(episode_path, 'obs')
        os.makedirs(self.obs_dir, exist_ok=True)


        # Load system prompt from file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.instruction_prompt_file_path = os.path.join(base_dir, instruction_prompt_file_path)
        with open(self.instruction_prompt_file_path, 'r') as f:
            self.system_prompt = f.read()
            self.system_prompt2 = f.read()

        # Add COT instruction if needed
        if COT:
            self.system_prompt2 += ("\nPlease explain your reasoning, then end with 'description: <your object coordinate list>',"
                                   "no matter what always end with description:<your object coordinate list> (dont add additional character"
                                   "after the word description)")
        else:
            self.system_prompt2 += "\nPlease output only the description, no explanations needed."


        # Add COT instruction if needed
        if COT:
            self.system_prompt += (
            """
                ## Explain Your Reasoning
                - Before every action, explain your reasoning clearly.
                - At the end of every response, include this line exactly:
                action: <your action>
                - Replace <your action> with the valid move action based on your reasoning.
                - Do not add any characters after the word action.
            """)
        else:
            self.system_prompt += "\nPlease output only the action, no explanations needed."

    def encode_image_from_pil(self, pil_image):
        """Convert PIL Image to base64 string for API consumption."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def process_goal_state(self, observation):
        """Process and store the goal state"""
        self.episode_logger.info("\n=== Receiving Goal State ===")
        self.goal_state = observation
        return "start"


    def parse_action(self, response, split_at="action: move "):
        """
        Extract the last action from model response, handling case-insensitivity,
        special characters, and replacing 2D terms with original 3D bodies.
        """

        # Define a mapping of 2D terms to original 3D bodies
        shape_map = {
            "circle": "sphere",
            "square": "cube",
            "triangle": "pyramid",
            "hexagon": "cylinder"
        }

        # Preprocess the response: Normalize spaces, remove line breaks, and clean unwanted characters
        cleaned_response = response.replace("\n", " ").replace("*", "").replace("#", "").strip().lower()

        # Replace 2D terms with their corresponding 3D bodies
        for shape_2d, body_3d in shape_map.items():
            cleaned_response = cleaned_response.replace(shape_2d, body_3d)

        # Find the last occurrence of the cleaned split string
        split_index = cleaned_response.rfind(split_at.lower())
        if split_index != -1:
            # Extract thoughts and action
            thoughts = cleaned_response[:split_index].strip()  # Extract text before the last occurrence
            action = "move " + cleaned_response[split_index + len(split_at):].strip()  # Extract text after the last occurrence
        else:
            # Handle case where split_at is not found
            logging.warning(f"Action format incorrect: {response}")
            self.episode_logger.warning(f"Action format incorrect: {response}")
            thoughts = response
            action = "No valid action found."

        return action, thoughts

    def parse_action_rmv_special_chars(self, action):
        """
        Cleans the input string to ensure it is safe to use as part of a file name.

        Args:
            action (str): The input string to parse and clean.

        Returns:
            str: The cleaned string, safe for use in file names.
        """
        # Remove invalid file name characters
        cleaned_action = re.sub(r'[\\/*?:"<>|]', '', action)

        # Replace newlines, tabs, and control characters with spaces
        cleaned_action = re.sub(r'[\r\n\t]', ' ', cleaned_action)

        # Remove other special characters (excluding alphanumeric and spaces)
        cleaned_action = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_action)

        # Remove leading backslashes or forward slashes
        cleaned_action = re.sub(r'^[\\/]+', '', cleaned_action)

        # Remove trailing single quotes or other special characters
        cleaned_action = cleaned_action.strip("'").strip()

        # Remove extra spaces
        cleaned_action = re.sub(r'\s+', ' ', cleaned_action).strip()

        return cleaned_action

    def remove_special_characters(self, input_string):
        """
        Remove special characters from a string, leaving only printable ASCII characters.

        Args:
            input_string (str): The input string to process.

        Returns:
            str: The string with special characters removed.
        """
        # Keep only printable ASCII characters (from space to ~)
        return re.sub(r'[^\x20-\x7E]', '->', input_string)


    def color_code_observation(self, observation, background_color):
        """
        Preprocesses an images by removing transparency and setting a solid background.
        """
        background = Image.new('RGB', observation.size, background_color)
        background.paste(observation, mask=observation.getchannel('A') if 'A' in observation.getbands() else None)
        return background

    def add_action_text(self, image, action_text, color="black"):
        """
        Adds the action text in large red letters with a light white, semi-transparent box behind it
        on the top-right corner of the image, 50 pixels away from the top and right border, with rounded corners.

        Args:
            image (PIL.Image): The image on which to add the action text.
            action_text (str): The action text to add to the image.

        Returns:
            PIL.Image: The image with the action text added.
        """
        # Ensure the image is in RGBA mode (supports transparency)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Try to load a larger font size
        try:
            font = ImageFont.truetype("arial.ttf", 50)  # Increased font size to 80
        except IOError:
            logging.error("Warning: 'arial.ttf' not found. Using default font.")
            font = ImageFont.load_default()

        # Get text bounding box
        text_bbox = draw.textbbox((0, 0), action_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Add extra padding to make the box slightly larger than the text
        box_extra_padding = 20  # Adjust this value to control how much larger the box should be
        border_thickness = 3  # Thickness of the black border

        # Set padding, border offset, and corner radius for rounded edges
        padding = 10
        border_offset = 50
        corner_radius = 20  # Radius for the rounded corners

        # Calculate box and text position (50 pixels from the top and right with additional padding)
        text_x = image.width - text_width - padding - border_offset
        text_y = border_offset
        box_x0 = text_x - padding - box_extra_padding
        box_y0 = text_y - padding - box_extra_padding
        box_x1 = image.width - padding - border_offset + box_extra_padding
        box_y1 = text_y + text_height + padding + box_extra_padding

        # Create a semi-transparent overlay
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))  # Transparent overlay
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw a black border (slightly larger rectangle)
        overlay_draw.rounded_rectangle(
            [box_x0 - border_thickness, box_y0 - border_thickness,
             box_x1 + border_thickness, box_y1 + border_thickness],
            radius=corner_radius + border_thickness,
            fill=(0, 0, 0, 255)  # Black border
        )

        # Draw a semi-transparent white box with rounded corners (50% transparency)
        overlay_draw.rounded_rectangle(
            [box_x0, box_y0, box_x1, box_y1],
            radius=corner_radius,
            fill=(255, 255, 255, 128)
        )

        # Composite the overlay onto the image
        image = Image.alpha_composite(image, overlay)

        # Draw the text in red on the original image
        draw = ImageDraw.Draw(image)
        draw.text((text_x, text_y), action_text, font=font, fill=color)

        return image

def load_api_keys(api_key_file_path):
    """Load API keys from file"""
    keys = {}
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    api_key_file_path = os.path.join(base_dir, api_key_file_path)
    with open(api_key_file_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                keys[key] = value
    return keys

## used for debugging claude and gpt
def format_message_structure(messages, cutoff=None):
    """
    Format message structure in a clean format, similar to:
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [IMAGE]},
        ...
    ]

    Args:
        messages (list): List of message dictionaries.
        cutoff (int, optional): Length at which to truncate text. If None, full text is shown.

    Returns:
        str: Formatted string representing the message structure.
    """
    result = ["\nmessages=["]
    for msg in messages:
        content = msg['content']

        # Handle different content types
        if isinstance(content, list):
            # For messages with image content
            if len(content) == 1 and content[0].get('type') in ['image', 'image_url']:
                content_preview = "[IMAGE]"
            else:
                # For messages with multiple content items
                content_preview = "["
                for item in content:
                    if item['type'] in ['image', 'image_url']:
                        content_preview += "[IMAGE], "
                    elif item['type'] == 'text':
                        text = item['text']
                        if cutoff and len(text) > cutoff:
                            content_preview += f"text: '{text[:cutoff]}...', "
                        else:
                            content_preview += f"text: '{text}', "
                content_preview = content_preview.rstrip(', ') + "]"
        elif isinstance(content, dict):
            # For single image messages
            if content.get('type') in ['image', 'image_url']:
                content_preview = "[IMAGE]"
            else:
                content_preview = str(content)
        else:
            # For simple text messages
            if cutoff and len(str(content)) > cutoff:
                content_preview = f"'{content[:cutoff]}...'"
            else:
                content_preview = f"'{content}'"

        result.append(f"    {{\"role\": \"{msg['role']}\", \"content\": {content_preview}}},")

    result.append("]")

    # Join the list into a single string with line breaks
    return "\n".join(result)


def format_message_structure_gemini(content, cutoff=None):
    """
    Format content structure in a clean format for Gemini API, similar to:
    content=[
        "instruction text...",
        [IMAGE],
        "Previous action taken: move_left",
        [IMAGE],
        ...
    ]

    Args:
        content (list): List of content items, including text and images.
        cutoff (int, optional): Length at which to truncate text. If None, full text is shown.

    Returns:
        str: Formatted string representing the content structure.
    """
    result = ["\ncontent=["]

    for item in content:
        # Handle different content types
        if isinstance(item, Image.Image):
            content_preview = "[IMAGE]"
        elif isinstance(item, str):
            if cutoff and len(item) > cutoff:
                content_preview = f"'{item[:cutoff]}...'"
            else:
                content_preview = f"'{item}'"
        else:
            content_preview = str(item)

        result.append(f"    {content_preview},")

    result.append("]")

    # Join the list into a single string with line breaks
    return "\n".join(result)


class GPT4Agent(LLMAgent):
    def __init__(self, episode_path, episode_logger, api_key_file_path, instruction_prompt_file_path, visual_state_embedding, single_images=True, COT=False, delay=0, max_history=0):
        super().__init__(episode_path, episode_logger,api_key_file_path, instruction_prompt_file_path, visual_state_embedding, single_images, COT, delay, max_history)
        self.api_key = self.api_keys['GPT4_API_KEY']
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.model = "gpt-4o"
        self.chat_history = []
        self.content = []  

    def act(self, observation, loop_iteration):
        sceneUnderstanding = False #TODO dirty quick code to manually change agent for scene understanding
        if sceneUnderstanding:
            try:
                self.episode_logger.info("\n=== Processing Current State ===")
                current_base64 = self.encode_image_from_pil(observation)
                RGB_blue = (82, 138, 174)  # "blue"
                current_base64_colored = self.color_code_observation(current_base64.copy(), RGB_blue)

                content = [
                    {"type": "text", "text": """In this image with blue background you will see the board, describe the board state. Only describe the objects with their coordinates, don't describe empty fields."""},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_base64_colored}"}}
                ]

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt2},
                        {"role": "user", "content": content}
                    ],
                    "max_tokens": 500
                }

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload
                )

                action, thoughts = self.parse_action(response.json()['choices'][0]['message']['content'], split_at='description:')
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nGPT suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nGPT suggested action: {action}")

                # Store the current exchange in history
                self.chat_history.append({
                    "observation": observation,
                    "response": action
                })

                return action


            except Exception as e:
                logging.error(f"\nError calling OpenAI API: {e}")
                return "error"


        elif isinstance(observation, Image.Image):
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                self.episode_logger.info("\n=== Processing Current State ===")

                # Block of code that embeds information what this image represents, e.g. active (current) state, goal state
                # or past state (later). There are several settings we can check for where the information about the
                # state the image represents is marked by color, label, both or none
                if self.visual_state_embedding == 'color':
                    current = self.color_code_observation(observation.copy(), tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(), tuple(self.color_codes['goal_1']['rgb']))
                elif self.visual_state_embedding == 'label':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['active_1']['rgb']))
                    current = self.add_action_text(current.copy(), 'active')
                    goal = self.add_action_text(goal.copy(), 'goal')
                elif self.visual_state_embedding == 'both':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['goal_1']['rgb']))
                    current = self.add_action_text(current.copy(), 'active')
                    goal = self.add_action_text(goal.copy(), 'goal')
                elif self.visual_state_embedding == 'none':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['active_1']['rgb']))


                current_base64 = self.encode_image_from_pil(current)
                goal_base64 = self.encode_image_from_pil(goal)

                messages = [{"role": "system", "content": self.system_prompt}]

                # Add history if available
                if self.max_history > 0:
                    for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):

                        # Block of code that embeds information what this image represents, here past state.
                        # There are several settings we can check for where the information about the
                        # state the image represents is marked by color, label, both or none
                        if self.visual_state_embedding == 'color':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(), tuple(self.color_codes['past']['rgb']))
                        elif self.visual_state_embedding == 'label':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),  tuple(self.color_codes['active_1']['rgb']))
                            prev_image_colored = self.add_action_text(prev_image_colored.copy(), 'past')
                        elif self.visual_state_embedding == 'both':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),tuple(self.color_codes['past']['rgb']))
                            prev_image_colored = self.add_action_text(prev_image_colored.copy(), 'past')
                        elif self.visual_state_embedding == 'none':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),  tuple(self.color_codes['active_1']['rgb']))


                        prev_image_base64 = self.encode_image_from_pil(prev_image_colored)
                        # User message with image
                        messages.append({
                            "role": "user", 
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_image_base64}"}}
                            ]
                        })
                        # Assistant response
                        messages.append({
                            "role": "assistant",
                            "content": f"Action: {prev_exchange['response']}"
                        })

                # Add goal state image
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{goal_base64}"}}
                    ]
                })

                # Add current state image
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_base64}"}}
                    ]
                })


                # Debug block that saves the images with the colors or text labels into debug_obs to check if
                # The code works correctly
                if DEBUG:
                    parent_dir = os.path.dirname(self.obs_dir)  # Get parent directory of obs_dir
                    debug_dir = os.path.join(parent_dir, 'debug_obs')  # Create debug_obs next to obs_dir

                    os.makedirs(debug_dir, exist_ok=True)

                    filename_obs_current = os.path.join(debug_dir, f"{loop_iteration}_current")
                    # Save the image as a PNG file
                    filepath_obs_current = f"{filename_obs_current}.png"
                    try:
                        current.save(filepath_obs_current)  # Save the image as a PNG
                    except Exception as e:
                        logging.error(f"Error saving image to {filename_obs_current}.png: {e}")

                    filename_obs_goal = os.path.join(debug_dir, f"{loop_iteration}_goal")
                    # Save the image as a PNG file
                    filepath_obs_goal = f"{filename_obs_goal}.png"
                    try:
                        goal.save(filepath_obs_goal)  # Save the image as a PNG
                    except Exception as e:
                        logging.error(f"Error saving image to {filename_obs_goal}.png: {e}")

                    filename_obs_past = os.path.join(debug_dir, f"{loop_iteration}_past")
                    # Save the image as a PNG file
                    filepath_obs_past = f"{filename_obs_past}.png"
                    if self.max_history > 0:
                        for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):
                            try:
                                prev_image_colored.save(filepath_obs_past)  # Save the image as a PNG
                            except Exception as e:
                                logging.error(f"Error saving image to {filename_obs_past}.png: {e}")

                    # Block of code that add to the prompt information what states the images represent,  e.g. active (current)
                    # state, goal state or past state. There are several settings we can check for where the
                    # information about the state the image represents is marked by color, label, both or none
                if self.visual_state_embedding == 'both':
                    text_snippet_active = f"marked with the label {self.color_codes['active_1']['type']} in the image and a {self.color_codes['active_1']['color']} background"
                    text_snippet_goal = f"marked with the label {self.color_codes['goal_1']['type']} in the image and a {self.color_codes['goal_1']['color']} background"
                    text_snippet_past = f"marked with the label {self.color_codes['past']['type']} in the image and a {self.color_codes['past']['color']} background"
                elif self.visual_state_embedding == 'color':
                    text_snippet_active = f" with a {self.color_codes['active_1']['color']} background"
                    text_snippet_goal = f" with a {self.color_codes['goal_1']['color']} background"
                    text_snippet_past = f" with a {self.color_codes['past']['color']} background"
                elif self.visual_state_embedding == 'label':
                    text_snippet_active = f"showing the label in the image {self.color_codes['active_1']['type']}"
                    text_snippet_goal = f"showing the label in the image {self.color_codes['goal_1']['type']}"
                    text_snippet_past = f"showing the label in the image {self.color_codes['past']['type']}"
                elif self.visual_state_embedding == 'none':
                    text_snippet_active = ""
                    text_snippet_goal = ""
                    text_snippet_past = ""
                else:
                    raise ValueError("No viable image state embedding set for agent.")

                # Add instruction text
                messages.append({
                    "role": "user",
                    "content": f"""
                        ## Analyze the Images
                        You can view your current active board state in the last image {text_snippet_active}. Study this image and the objects with their positions carefully.
                        Your goal is to match the goal state, shown in the image {text_snippet_goal}. Study this image and the objects with their positions carefully.

                        ## Additionally, you are provided with:
                        - The previous state image(s) {text_snippet_past}.
                        - Your previous suggested action
                        - Use this information by comparing it to your current active state to determine your next action.

                        ## Invalid Actions:
                        - No Overlap: You are not allowed to position two objects in the same tile.
                        - If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
                        - Use the previous image(s) and action to understand why it failed and suggest a different action.

                        It is of most importance you always end your response with this exact format:

                        action: move <object color> <object shape> <direction>
                        where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
                """
                })

                formatted_message = format_message_structure(messages) # use it for debugging make sure the input is correct
                self.episode_logger.info(formatted_message)

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 500
                }

                response = requests.post("https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload
                )

                action, thoughts = self.parse_action(response.json()['choices'][0]['message']['content'])
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nGPT suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nGPT suggested action: {action}")

                # Store the current exchange in history
                self.chat_history.append({
                    "observation": observation,
                    "response": action
                })

                return action

            except Exception as e:
                logging.error(f"\nError calling OpenAI API: {e}")
                return "error"

        elif isinstance(observation, list) and all(isinstance(item, str) for item in observation):
            observation = '\n'.join(observation)

            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                self.episode_logger.info("\n=== Processing Current State ===")

                # Build messages with history
                messages = [{"role": "system", "content": self.system_prompt}]
                
                # Add conversation history
                if self.max_history > 0:
                    for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):
                        messages.extend([
                            {"role": "user", "content": f"Previous observation {i}: {prev_exchange['observation']}"},
                            {"role": "assistant", "content": f"Previous action {i}: {prev_exchange['response']}"}
                        ])

                # Add current observation
                current_content = [
                    {"type": "text", "text": f"""
                        ## Analyze the States
                        The goal state is provided as the first set of coordinates: 
                        {self.goal_state}.
                        The current state is provided as the second set of coordinates: 
                        {observation}.
                        Study both states carefully and determine how to move objects in the current state to match the goal state.
                        
                        ## Additionally, you are provided with:
                        - The previous states.
                        - Your previous suggested action
                        - Use this information by comparing it to your current active state to determine your next action.
                        
                        ## Invalid Actions:
                        - No Overlap: You are not allowed to position two objects in the same tile.
                        - If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
                        - Use the previous image(s) and action to understand why it failed and suggest a different action.
                        
                        It is of most importance you always end your response with this exact format:
                        
                        action: move <object color> <object shape> <direction>
                        where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
                    """},]
                messages.append({"role": "user", "content": current_content})

                formatted_message = format_message_structure(messages)
                self.episode_logger.info(formatted_message)

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.5 # default is 0
                }

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload
                )

                action, thoughts = self.parse_action(response.json()['choices'][0]['message']['content'])
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nGPT suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nGPT suggested action: {action}")

                # Store the exchange in history
                self.chat_history.append({
                    "observation": observation,
                    "response": action
                })

                return action

            except Exception as e:
                logging.error(f"\nError calling OpenAI API: {e}")
                return "error"


class ClaudeAgent(LLMAgent):

    def __init__(self, episode_path, episode_logger, api_key_file_path, instruction_prompt_file_path, visual_state_embedding, single_images=True, COT=False, delay=0, max_history=0):
        super().__init__(episode_path, episode_logger, api_key_file_path, instruction_prompt_file_path, visual_state_embedding, single_images, COT, delay, max_history)
        self.client = Anthropic(api_key=self.api_keys['CLAUDE_API_KEY'])
        self.model = "claude-3-5-sonnet-20241022"
        self.chat_history = []
        self.content = []  
       
    def act(self, observation, loop_iteration):
        if isinstance(observation, Image.Image):
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                self.episode_logger.info("\n=== Processing Current State ===")


                # Block of code that embeds information what this image represents, e.g. active (current) state, goal state
                # or past state (later). There are several settings we can check for where the information about the
                # state the image represents is marked by color, label, both or none
                if self.visual_state_embedding == 'color':
                    current = self.color_code_observation(observation.copy(), tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(), tuple(self.color_codes['goal_1']['rgb']))
                elif self.visual_state_embedding == 'label':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['active_1']['rgb']))
                    current = self.add_action_text(current.copy(), 'active')
                    goal = self.add_action_text(goal.copy(), 'goal')
                elif self.visual_state_embedding == 'both':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['goal_1']['rgb']))
                    current = self.add_action_text(current.copy(), 'active')
                    goal = self.add_action_text(goal.copy(), 'goal')
                elif self.visual_state_embedding == 'none':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['active_1']['rgb']))


                current_base64 = self.encode_image_from_pil(current)
                goal_base64 = self.encode_image_from_pil(goal)

                messages = []

                # Add history if available
                if self.max_history > 0:
                    for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):

                        # Block of code that embeds information what this image represents, here past state.
                        # There are several settings we can check for where the information about the
                        # state the image represents is marked by color, label, both or none
                        if self.visual_state_embedding == 'color':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(), tuple(self.color_codes['past']['rgb']))
                        elif self.visual_state_embedding == 'label':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),  tuple(self.color_codes['active_1']['rgb']))
                            prev_image_colored = self.add_action_text(prev_image_colored.copy(), 'past')
                        elif self.visual_state_embedding == 'both':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),tuple(self.color_codes['past']['rgb']))
                            prev_image_colored = self.add_action_text(prev_image_colored.copy(), 'past')
                        elif self.visual_state_embedding == 'none':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),  tuple(self.color_codes['active_1']['rgb']))

                        prev_image_base64 = self.encode_image_from_pil(prev_image_colored)

                        # Add previous image
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": prev_image_base64}}
                            ]
                        })
                        # Add assistant's response
                        messages.append({
                            "role": "assistant",
                            "content": f"Action: {prev_exchange['response']}"
                        })


                # Add goal state image
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": goal_base64}}
                    ]
                })

                # Add current state image
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": current_base64}}
                    ]
                })

                # Block of code that add to the prompt information what states the images represent,  e.g. active (current)
                # state, goal state or past state. There are several settings we can check for where the
                # information about the state the image represents is marked by color, label, both or none
                if self.visual_state_embedding == 'both':
                    text_snippet_active = f"marked with the label {self.color_codes['active_1']['type']} in the image and a {self.color_codes['active_1']['color']} background"
                    text_snippet_goal = f"marked with the label {self.color_codes['goal_1']['type']} in the image and a {self.color_codes['goal_1']['color']} background"
                    text_snippet_past = f"marked with the label {self.color_codes['past']['type']} in the image and a {self.color_codes['past']['color']} background"
                elif self.visual_state_embedding == 'color':
                    text_snippet_active = f" with a {self.color_codes['active_1']['color']} background"
                    text_snippet_goal = f" with a {self.color_codes['goal_1']['color']} background"
                    text_snippet_past = f" with a {self.color_codes['past']['color']} background"
                elif self.visual_state_embedding == 'label':
                    text_snippet_active = f"showing the label in the image {self.color_codes['active_1']['type']}"
                    text_snippet_goal = f"showing the label in the image {self.color_codes['goal_1']['type']}"
                    text_snippet_past = f"showing the label in the image {self.color_codes['past']['type']}"
                elif self.visual_state_embedding == 'none':
                    text_snippet_active = ""
                    text_snippet_goal = ""
                    text_snippet_past = ""
                else:
                    raise ValueError("No viable image state embedding set for agent.")

                # Add instruction text
                messages.append({
                    "role": "user",
                    "content": f"""
                        ## Analyze the Images
                        You can view your current active board state in the last image {text_snippet_active}. Study this image and the objects with their positions carefully.
                        Your goal is to match the goal state, shown in the image {text_snippet_goal}. Study this image and the objects with their positions carefully.
                        
                        ## Additionally, you are provided with:
                        - The previous state image(s) {text_snippet_past}.
                        - Your previous suggested action
                        - Use this information by comparing it to your current active state to determine your next action.
                        
                        ## Invalid Actions:
                        - No Overlap: You are not allowed to position two objects in the same tile.
                        - If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
                        - Use the previous image(s) and action to understand why it failed and suggest a different action.
                        
                        It is of most importance you always end your response with this exact format:
                        
                        action: move <object color> <object shape> <direction>
                        where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
                """
                })


                # Debug block that saves the images with the colors or text labels into debug_obs to check if
                # The code works correctly
                if DEBUG:
                    parent_dir = os.path.dirname(self.obs_dir)  # Get parent directory of obs_dir
                    debug_dir = os.path.join(parent_dir, 'debug_obs')  # Create debug_obs next to obs_dir

                    os.makedirs(debug_dir, exist_ok=True)

                    filename_obs_current = os.path.join(debug_dir, f"{loop_iteration}_current")
                    # Save the image as a PNG file
                    filepath_obs_current = f"{filename_obs_current}.png"
                    try:
                        current.save(filepath_obs_current)  # Save the image as a PNG
                    except Exception as e:
                        logging.error(f"Error saving image to {filename_obs_current}.png: {e}")

                    filename_obs_goal = os.path.join(debug_dir, f"{loop_iteration}_goal")
                    # Save the image as a PNG file
                    filepath_obs_goal = f"{filename_obs_goal}.png"
                    try:
                        goal.save(filepath_obs_goal)  # Save the image as a PNG
                    except Exception as e:
                        logging.error(f"Error saving image to {filename_obs_goal}.png: {e}")

                    filename_obs_past = os.path.join(debug_dir, f"{loop_iteration}_past")
                    # Save the image as a PNG file
                    filepath_obs_past = f"{filename_obs_past}.png"
                    if self.max_history > 0:
                        for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):
                            try:
                                prev_image_colored.save(filepath_obs_past)  # Save the image as a PNG
                            except Exception as e:
                                logging.error(f"Error saving image to {filename_obs_past}.png: {e}")


                formatted_message = format_message_structure(messages)
                self.episode_logger.info(formatted_message)

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    system=self.system_prompt,
                    temperature=0.5,
                    messages=messages
                )

                action, thoughts = self.parse_action(message.content[0].text)
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nClaude suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nClaude suggested action: {action}")

                # Store the current exchange in history
                self.chat_history.append({
                    "observation": observation,
                    "response": action
                })

                return action

            except Exception as e:
                logging.error(f"\nError calling Claude API: {e}")
                return "error"

        elif isinstance(observation, list) and all(isinstance(item, str) for item in observation):
            
            observation = '\n'.join(observation)
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)  
                self.episode_logger.info("\n=== Processing Current State ===")
                messages = []


                if self.max_history > 0:
                    for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):
                        messages.extend([
                            {"role": "user", "content": f"Previous observation {i}: {prev_exchange['observation']}"},
                            {"role": "assistant", "content": f"Previous action {i}: {prev_exchange['response']}"}
                        ])
                current_content = [
                    {"type": "text", "text": f"""
                        ## Analyze the States
                        The goal state is provided as the first set of coordinates: 
                        {self.goal_state}.
                        The current state is provided as the second set of coordinates: 
                        {observation}.
                        Study both states carefully and determine how to move objects in the current state to match the goal state.

                        ## Additionally, you are provided with:
                        - The previous states.
                        - Your previous suggested action
                        - Use this information by comparing it to your current active state to determine your next action.

                        ## Invalid Actions:
                        - No Overlap: You are not allowed to position two objects in the same tile.
                        - If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
                        - Use the previous image(s) and action to understand why it failed and suggest a different action.

                        It is of most importance you always end your response with this exact format:

                        action: move <object color> <object shape> <direction>
                        where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
                    """}, ]

                messages.append({"role": "user", "content": current_content})

                formatted_message = format_message_structure(messages)
                self.episode_logger.info(formatted_message)


                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    system=self.system_prompt,
                    temperature=0.5,  # maybe needs too be tuned
                    messages= messages
                )

                action, thoughts = self.parse_action(message.content[0].text)
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nClaude suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nClaude suggested action: {action}")

                self.chat_history.append({
                    "observation": observation,
                    "response": action
                })

                return action

            except Exception as e:
                logging.error(f"\nError calling Claude API: {e}")
                return "error"
            
class GeminiAgent(LLMAgent):
    def __init__(self, episode_path, episode_logger, api_key_file_path, instruction_prompt_file_path, visual_state_embedding, single_images=True, COT=False, delay=0, max_history=0):
        super().__init__(episode_path, episode_logger, api_key_file_path, instruction_prompt_file_path, visual_state_embedding, single_images, COT, delay, max_history)
        self.api_key = self.api_keys['GEMINI_API_KEY']
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-exp", system_instruction=self.system_prompt)
        self.chat_history = []
        self.content = [] 

    def act(self, observation, loop_iteration):
        if isinstance(observation, Image.Image):
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                self.episode_logger.info("\n=== Processing Current State ===")
                
                # Initialize content list for Gemini
                messages = []
                
                # Add history if available
                if self.max_history > 0:
                    for prev_exchange in self.chat_history[-self.max_history:]:
                        # Color the previous state orange and add to messages


                        # Block of code that embeds information what this image represents, here past state.
                        # There are several settings we can check for where the information about the
                        # state the image represents is marked by color, label, both or none
                        if self.visual_state_embedding == 'color':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(), tuple(self.color_codes['past']['rgb']))
                        elif self.visual_state_embedding == 'label':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),  tuple(self.color_codes['active_1']['rgb']))
                            prev_image_colored = self.add_action_text(prev_image_colored.copy(), 'past')
                        elif self.visual_state_embedding == 'both':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),tuple(self.color_codes['past']['rgb']))
                            prev_image_colored = self.add_action_text(prev_image_colored.copy(), 'past')
                        elif self.visual_state_embedding == 'none':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),  tuple(self.color_codes['active_1']['rgb']))

                        messages.extend([
                            prev_image_colored,
                            f"Previous action: {prev_exchange['response']}"
                        ])

                # Block of code that embeds information what this image represents, e.g. active (current) state, goal state
                # or past state (above). There are several settings we can check for where the information about the
                # state the image represents is marked by color, label, both or none
                if self.visual_state_embedding == 'color':
                    current = self.color_code_observation(observation.copy(), tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(), tuple(self.color_codes['goal_1']['rgb']))
                elif self.visual_state_embedding == 'label':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['active_1']['rgb']))
                    current = self.add_action_text(current.copy(), 'active')
                    goal = self.add_action_text(goal.copy(), 'goal')
                elif self.visual_state_embedding == 'both':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['goal_1']['rgb']))
                    current = self.add_action_text(current.copy(), 'active')
                    goal = self.add_action_text(goal.copy(), 'goal')
                elif self.visual_state_embedding == 'none':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['active_1']['rgb']))


                messages.extend([
                    goal,    # Goal state image (green)
                    current, # Current state image (blue)
                ])

                # Block of code that add to the prompt information what states the images represent,  e.g. active (current)
                # state, goal state or past state. There are several settings we can check for where the
                # information about the state the image represents is marked by color, label, both or none
                if self.visual_state_embedding == 'both':
                    text_snippet_active = f"marked with the label {self.color_codes['active_1']['type']} in the image and a {self.color_codes['active_1']['color']} background"
                    text_snippet_goal = f"marked with the label {self.color_codes['goal_1']['type']} in the image and a {self.color_codes['goal_1']['color']} background"
                    text_snippet_past = f"marked with the label {self.color_codes['past']['type']} in the image and a {self.color_codes['past']['color']} background"
                elif self.visual_state_embedding == 'color':
                    text_snippet_active = f" with a {self.color_codes['active_1']['color']} background"
                    text_snippet_goal = f" with a {self.color_codes['goal_1']['color']} background"
                    text_snippet_past = f" with a {self.color_codes['past']['color']} background"
                elif self.visual_state_embedding == 'label':
                    text_snippet_active = f"showing the label in the image {self.color_codes['active_1']['type']}"
                    text_snippet_goal = f"showing the label in the image {self.color_codes['goal_1']['type']}"
                    text_snippet_past = f"showing the label in the image {self.color_codes['past']['type']}"
                elif self.visual_state_embedding == 'none':
                    text_snippet_active = ""
                    text_snippet_goal = ""
                    text_snippet_past = ""
                else:
                    raise ValueError("No viable image state embedding set for agent.")

                # Add instruction text
                instruction = f"""
                    ## Analyze the Images
                    You can view your current active board state in the last image {text_snippet_active}. Study this image and the objects with their positions carefully.
                    Your goal is to match the goal state, shown in the image {text_snippet_goal}. Study this image and the objects with their positions carefully.
                    
                    ## Additionally, you are provided with:
                    - The previous state image(s) {text_snippet_past}.
                    - Your previous suggested action
                    - Use this information by comparing it to your current active state to determine your next action.
                    
                    ## Invalid Actions:
                    - No Overlap: You are not allowed to position two objects in the same tile.
                    - If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
                    - Use the previous image(s) and action to understand why it failed and suggest a different action.
                    
                    It is of most importance you always end your response with this exact format:
                    
                    action: move <object color> <object shape> <direction>
                    where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
                """
            
                messages.extend([instruction])
                formatted_message = format_message_structure_gemini(messages) # use it for debugging make sure the input is correct
                self.episode_logger.info(formatted_message)


                # Debug block that saves the images with the colors or text labels into debug_obs to check if
                # The code works correctly
                if DEBUG:
                    parent_dir = os.path.dirname(self.obs_dir)  # Get parent directory of obs_dir
                    debug_dir = os.path.join(parent_dir, 'debug_obs')  # Create debug_obs next to obs_dir

                    os.makedirs(debug_dir, exist_ok=True)

                    filename_obs_current = os.path.join(debug_dir, f"{loop_iteration}_current")
                    # Save the image as a PNG file
                    filepath_obs_current = f"{filename_obs_current}.png"
                    try:
                        current.save(filepath_obs_current)  # Save the image as a PNG
                    except Exception as e:
                        logging.error(f"Error saving image to {filename_obs_current}.png: {e}")

                    filename_obs_goal = os.path.join(debug_dir, f"{loop_iteration}_goal")
                    # Save the image as a PNG file
                    filepath_obs_goal = f"{filename_obs_goal}.png"
                    try:
                        goal.save(filepath_obs_goal)  # Save the image as a PNG
                    except Exception as e:
                        logging.error(f"Error saving image to {filename_obs_goal}.png: {e}")

                    filename_obs_past = os.path.join(debug_dir, f"{loop_iteration}_past")
                    # Save the image as a PNG file
                    filepath_obs_past = f"{filename_obs_past}.png"
                    if self.max_history > 0:
                        for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):
                            try:
                                prev_image_colored.save(filepath_obs_past)  # Save the image as a PNG
                            except Exception as e:
                                logging.error(f"Error saving image to {filename_obs_past}.png: {e}")



                # Generate response using Gemini
                response = self.model.generate_content(
                    messages,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=500
                    )
                )

                action, thoughts = self.parse_action(response.text)
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nGemini suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nGemini suggested action: {action}")

                
                # Store the current exchange in history
                self.chat_history.append({
                    "observation": observation,
                    "response": action
                })

                return action

            except Exception as e:
                logging.error(f"\nError calling Gemini API: {e}")
                return "error"
            
        elif isinstance(observation, list) and all(isinstance(item, str) for item in observation):
            
            observation = '\n'.join(observation)
            try:
                if self.goal_state is None:
                    return self.process_goal_state(observation)  
                self.episode_logger.info("\n=== Processing Current State ===")

                if self.max_history > 0:
                    history_context = ""
                    for prev_exchange in self.chat_history[-self.max_history:]:
                        history_context += f"Previous observation: {prev_exchange['observation']}\n"
                        history_context += f"Previous action: {prev_exchange['response']}\n\n"
                
                content = [f"""{history_context if self.max_history > 0 else ''}
                    The first coordinates show the goal state, and the second coordinates show
                    the current state. What action should be taken (Keep in mind, you are not allowed to have two objects in the same position)?
                    The board is 4 by 4, coordinates go from a to d and 1 to 4 but not beyond that.
                    goal state: {self.goal_state} 
                    current state: {observation}
                    no matter what always end with action: <your action> (dont add additional character
                    after the word action) """
                ]

                content = [
                    f"""
                        {history_context if self.max_history > 0 else ''}
                    
                        ## Analyze the States
                        The goal state is provided as the first set of coordinates: 
                        {self.goal_state}.
                        The current state is provided as the second set of coordinates: 
                        {observation}.
                        Study both states carefully and determine how to move objects in the current state to match the goal state.
    
                        ## Additionally, you are provided with:
                        - The previous states.
                        - Your previous suggested action
                        - Use this information by comparing it to your current active state to determine your next action.
    
                        ## Invalid Actions:
                        - No Overlap: You are not allowed to position two objects in the same tile.
                        - If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
                        - Use the previous image(s) and action to understand why it failed and suggest a different action.
    
                        It is of most importance you always end your response with this exact format:
    
                        action: move <object color> <object shape> <direction>
                        where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
                    """
                ]

                response = self.model.generate_content(content)
                action, thoughts = self.parse_action(response.text)
                thoughts = self.parse_action_rmv_special_chars(thoughts)
                action = self.parse_action_rmv_special_chars(action)
                self.episode_logger.info(f"\nGemini suggested thoughts: {thoughts}")
                self.episode_logger.info(f"\nGemini suggested action: {action}")

                self.chat_history.append({
                    "observation": observation,
                    "response": action
                })

                return action

            except Exception as e:
                logging.error(f"\nError calling Gemini API: {e}")
                return "error"

class OpenSourceAgent:
    def __init__(self, model_path, instruction_prompt_file_path, max_history, visual_state_embedding, single_images=True, COT=False):
        self.model_path = model_path
        self.model = None
        self.goal_state = None
        self.single_images = single_images
        self.COT = COT
        self.max_history = max_history
        self.visual_state_embedding = visual_state_embedding
        self.color_codes = util.load_params_from_json('color_codes.json')
        self.delay = 0.0

        # Load system prompt from file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.instruction_prompt_file_path = os.path.join(base_dir, instruction_prompt_file_path)
        with open(self.instruction_prompt_file_path, 'r') as f:
            self.system_prompt = f.read()
            self.system_prompt2 = f.read()


        # Add COT instruction if needed
        if COT:
            self.system_prompt += (
            """
                ## Explain Your Reasoning
                - Before every action, explain your reasoning clearly.
                - At the end of every response, include this line exactly:
                action: <your action>
                - Replace <your action> with the valid move action based on your reasoning.
                - Do not add any characters after the word action.
            """)
        else:
            self.system_prompt += "\nPlease output only the action, no explanations needed."
            
    def reset(self, episode_path, episode_logger):
        self.goal_state = None
        self.episode_logger = episode_logger
        
        # Create the 'obs' subdirectory inside the save path
        self.obs_dir = os.path.join(episode_path, 'obs')
        os.makedirs(self.obs_dir, exist_ok=True)
        
    def encode_image_from_pil(self, pil_image):
        """Convert PIL Image to base64 string for API consumption."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def process_goal_state(self, observation):
        """Process and store the goal state"""
        self.episode_logger.info("\n=== Receiving Goal State ===")
        self.goal_state = observation
        return "start"

    def parse_action(self, response, split_at="action: move "):
        """Extract the last action from model response."""
        # Find the last occurrence of the split string
        split_index = response.rfind(split_at)
        if split_index != -1:
            thoughts = response[:split_index].strip()  # Extract text before the last occurrence
            action = "move "+response[split_index + len(split_at):].strip()  # Extract text after the last occurrence
        else:
            # Handle case where split_at is not found
            logging.warning(f"Action format incorrect: {response}")
            self.episode_logger.warning(f"Action format incorrect: {response}")
            thoughts = response
            action = "No valid action found."
        return action, thoughts

    def parse_action_rmv_special_chars(self, action):
        """
        Cleans the input string to ensure it is safe to use as part of a file name.

        Args:
            action (str): The input string to parse and clean.

        Returns:
            str: The cleaned string, safe for use in file names.
        """
        # Remove invalid file name characters
        cleaned_action = re.sub(r'[\\/*?:"<>|]', '', action)

        # Replace newlines, tabs, and control characters with spaces
        cleaned_action = re.sub(r'[\r\n\t]', ' ', cleaned_action)

        # Remove other special characters (excluding alphanumeric and spaces)
        cleaned_action = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_action)

        # Remove leading backslashes or forward slashes
        cleaned_action = re.sub(r'^[\\/]+', '', cleaned_action)

        # Remove trailing single quotes or other special characters
        cleaned_action = cleaned_action.strip("'").strip()

        # Remove extra spaces
        cleaned_action = re.sub(r'\s+', ' ', cleaned_action).strip()

        return cleaned_action
    
    def remove_special_characters(self, input_string):
        """
        Remove special characters from a string, leaving only printable ASCII characters.
        Args:
            input_string (str): The input string to process.
        Returns:
            str: The string with special characters removed.
        """
        # Keep only printable ASCII characters (from space to ~)
        return re.sub(r'[^\x20-\x7E]', '->', input_string)
    
    def color_code_observation(self, observation, background_color):
        """
        Preprocesses an images by removing transparency and setting a solid background.
        """
        background = Image.new('RGB', observation.size, background_color)
        background.paste(observation, mask=observation.getchannel('A') if 'A' in observation.getbands() else None)
        return background

    def add_action_text(self, image, action_text, color="black"):
        """
        Adds the action text in large red letters with a light white, semi-transparent box behind it
        on the top-right corner of the image, 50 pixels away from the top and right border, with rounded corners.

        Args:
            image (PIL.Image): The image on which to add the action text.
            action_text (str): The action text to add to the image.

        Returns:
            PIL.Image: The image with the action text added.
        """
        # Ensure the image is in RGBA mode (supports transparency)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Try to load a larger font size
        try:
            font = ImageFont.truetype("arial.ttf", 50)  # Increased font size to 80
        except IOError:
            logging.error("Warning: 'arial.ttf' not found. Using default font.")
            font = ImageFont.load_default()

        # Get text bounding box
        text_bbox = draw.textbbox((0, 0), action_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Add extra padding to make the box slightly larger than the text
        box_extra_padding = 20  # Adjust this value to control how much larger the box should be
        border_thickness = 3  # Thickness of the black border

        # Set padding, border offset, and corner radius for rounded edges
        padding = 10
        border_offset = 50
        corner_radius = 20  # Radius for the rounded corners

        # Calculate box and text position (50 pixels from the top and right with additional padding)
        text_x = image.width - text_width - padding - border_offset
        text_y = border_offset
        box_x0 = text_x - padding - box_extra_padding
        box_y0 = text_y - padding - box_extra_padding
        box_x1 = image.width - padding - border_offset + box_extra_padding
        box_y1 = text_y + text_height + padding + box_extra_padding

        # Create a semi-transparent overlay
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))  # Transparent overlay
        overlay_draw = ImageDraw.Draw(overlay)

        # Draw a black border (slightly larger rectangle)
        overlay_draw.rounded_rectangle(
            [box_x0 - border_thickness, box_y0 - border_thickness,
             box_x1 + border_thickness, box_y1 + border_thickness],
            radius=corner_radius + border_thickness,
            fill=(0, 0, 0, 255)  # Black border
        )

        # Draw a semi-transparent white box with rounded corners (50% transparency)
        overlay_draw.rounded_rectangle(
            [box_x0, box_y0, box_x1, box_y1],
            radius=corner_radius,
            fill=(255, 255, 255, 128)
        )

        # Composite the overlay onto the image
        image = Image.alpha_composite(image, overlay)

        # Draw the text in red on the original image
        draw = ImageDraw.Draw(image)
        draw.text((text_x, text_y), action_text, font=font, fill=color)

        return image
    
    def act(self, observation, loop_iteration=None):
        raise NotImplementedError 


class Qwen2Agent(OpenSourceAgent):
    """Parent class for LLM-based agents"""

    def __init__(self, model_path, instruction_prompt_file_path, max_history, visual_state_embedding, single_images=True, COT=False):
        super().__init__(model_path, instruction_prompt_file_path, max_history, visual_state_embedding, single_images, COT)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", cache_dir='weights'
        )
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir='weights')
        
    def reset(self, episode_path, episode_logger):
        super().reset(episode_path, episode_logger)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.chat_history = []  

    def act(self, observation, loop_iteration=None):
        try:
            if isinstance(observation, Image.Image):
                if self.goal_state is None:
                    return self.process_goal_state(observation)

                self.episode_logger.info("\n=== Processing Current State ===")
                
                # Block of code that embeds information what this image represents, e.g. active (current) state, goal state
                # or past state (later). There are several settings we can check for where the information about the
                # state the image represents is marked by color, label, both or none
                if self.visual_state_embedding == 'color':
                    current = self.color_code_observation(observation.copy(), tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(), tuple(self.color_codes['goal_1']['rgb']))
                elif self.visual_state_embedding == 'label':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['active_1']['rgb']))
                    current = self.add_action_text(current.copy(), 'active')
                    goal = self.add_action_text(goal.copy(), 'goal')
                elif self.visual_state_embedding == 'both':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['goal_1']['rgb']))
                    current = self.add_action_text(current.copy(), 'active')
                    goal = self.add_action_text(goal.copy(), 'goal')
                elif self.visual_state_embedding == 'none':
                    current = self.color_code_observation(observation.copy(),tuple(self.color_codes['active_1']['rgb']))
                    goal = self.color_code_observation(self.goal_state.copy(),tuple(self.color_codes['active_1']['rgb']))
                    
                self.messages = [{"role": "system", "content": self.system_prompt}]
                
                current_base64 = self.encode_image_from_pil(current)
                goal_base64 = self.encode_image_from_pil(goal)
                
                # Add history if available
                if self.max_history > 0:
                    for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):

                        # Block of code that embeds information what this image represents, here past state.
                        # There are several settings we can check for where the information about the
                        # state the image represents is marked by color, label, both or none
                        if self.visual_state_embedding == 'color':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(), tuple(self.color_codes['past']['rgb']))
                        elif self.visual_state_embedding == 'label':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),  tuple(self.color_codes['active_1']['rgb']))
                            prev_image_colored = self.add_action_text(prev_image_colored.copy(), 'past')
                        elif self.visual_state_embedding == 'both':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),tuple(self.color_codes['past']['rgb']))
                            prev_image_colored = self.add_action_text(prev_image_colored.copy(), 'past')
                        elif self.visual_state_embedding == 'none':
                            prev_image_colored = self.color_code_observation(prev_exchange['observation'].copy(),  tuple(self.color_codes['active_1']['rgb']))

                        prev_image_base64 = self.encode_image_from_pil(prev_image_colored)

                        # Add previous image
                        self.messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f'data:image;base64,{prev_image_base64}'}
                            ],
                        })
                        
                        # Add assistant's response
                        self.messages.append({
                            "role": "assistant",
                            "content": f"Action: {prev_exchange['response']}"
                        })
                        
                        
                # Add goal state image
                self.messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f'data:image;base64,{goal_base64}'}
                    ]
                })

                # Add current state image
                self.messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f'data:image;base64,{current_base64}'}
                    ]
                })
                
                # Block of code that add to the prompt information what states the images represent,  e.g. active (current)
                # state, goal state or past state. There are several settings we can check for where the
                # information about the state the image represents is marked by color, label, both or none
                if self.visual_state_embedding == 'both':
                    text_snippet_active = f"marked with the label {self.color_codes['active_1']['type']} in the image and a {self.color_codes['active_1']['color']} background"
                    text_snippet_goal = f"marked with the label {self.color_codes['goal_1']['type']} in the image and a {self.color_codes['goal_1']['color']} background"
                    text_snippet_past = f"marked with the label {self.color_codes['past']['type']} in the image and a {self.color_codes['past']['color']} background"
                elif self.visual_state_embedding == 'color':
                    text_snippet_active = f" with a {self.color_codes['active_1']['color']} background"
                    text_snippet_goal = f" with a {self.color_codes['goal_1']['color']} background"
                    text_snippet_past = f" with a {self.color_codes['past']['color']} background"
                elif self.visual_state_embedding == 'label':
                    text_snippet_active = f"showing the label in the image {self.color_codes['active_1']['type']}"
                    text_snippet_goal = f"showing the label in the image {self.color_codes['goal_1']['type']}"
                    text_snippet_past = f"showing the label in the image {self.color_codes['past']['type']}"
                elif self.visual_state_embedding == 'none':
                    text_snippet_active = ""
                    text_snippet_goal = ""
                    text_snippet_past = ""
                else:
                    raise ValueError("No viable image state embedding set for agent.")

                # Add instruction text
                self.messages.append({
                    "role": "user",
                    "content": f"""
                        ## Analyze the Images
                        You can view your current active board state in the last image {text_snippet_active}. Study this image and the objects with their positions carefully.
                        Your goal is to match the goal state, shown in the image {text_snippet_goal}. Study this image and the objects with their positions carefully.
                        
                        ## Additionally, you are provided with:
                        - The previous state image(s) {text_snippet_past}.
                        - Your previous suggested action
                        - Use this information by comparing it to your current active state to determine your next action.
                        
                        ## Invalid Actions:
                        - No Overlap: You are not allowed to position two objects in the same tile.
                        - If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
                        - Use the previous image(s) and action to understand why it failed and suggest a different action.
                        
                        It is of most importance you always end your response with this exact format:
                        
                        action: move <object color> <object shape> <direction>
                        where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
                """
                })
            elif isinstance(observation, list) and all(isinstance(item, str) for item in observation):
                observation = '\n'.join(observation)
                if self.goal_state is None:
                    return self.process_goal_state(observation)
                self.episode_logger.info("\n=== Processing Current State ===")
                
                self.messages = [{"role": "system", "content": self.system_prompt}]
                
                if self.max_history > 0:
                    for i, prev_exchange in enumerate(self.chat_history[-self.max_history:], 1):
                        self.messages.extend([
                            {"role": "user", "content": f"Previous observation {i}: {prev_exchange['observation']}"},
                            {"role": "assistant", "content": f"Previous action {i}: {prev_exchange['response']}"}
                        ])
                        
                self.messages.append({"role": "user", "content": f"""
                        ## Analyze the States
                        The goal state is provided as the first set of coordinates: 
                        {self.goal_state}.
                        The current state is provided as the second set of coordinates: 
                        {observation}.
                        Study both states carefully and determine how to move objects in the current state to match the goal state.

                        ## Additionally, you are provided with:
                        - The previous states.
                        - Your previous suggested action
                        - Use this information by comparing it to your current active state to determine your next action.

                        ## Invalid Actions:
                        - No Overlap: You are not allowed to position two objects in the same tile.
                        - If the suggested action does not move any objects, it is invalid (e.g., blocked by another object or out of bounds).
                        - Use the previous image(s) and action to understand why it failed and suggest a different action.

                        It is of most importance you always end your response with this exact format:

                        action: move <object color> <object shape> <direction>
                        where you replace <object color> <object shape> <direction> with the valid move action based on your reasoning and do not add any characters after your action.
                    """})

            
            text = self.processor.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(self.messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=300, temperature=1.0,  # Increased from 0.7
                                   top_k=50,         # Added top_k sampling
                                   top_p=0.95,       # Added nucleus sampling
                                   do_sample=True ,   # Enable sampling
                                   repetition_penalty=1.05,)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            self.episode_logger.info(response)
            action, thoughts = self.parse_action(response)
            
            self.chat_history.append({
                "observation": observation,
                "response": action
            })
            
            self.episode_logger.info(f"\nQwen2 suggested action: {action}")
            self.episode_logger.info(f"\nQwen2 suggested thoughts: {thoughts}")
            action = self.parse_action_rmv_special_chars(action)
            return action
        
        except Exception as e:
            logging.error(f"\nError calling Gemini API: {e}")
            return "error"
      
class Qwen2_72BAgent(Qwen2Agent):
    def __init__(self, instruction_prompt_file_path, max_history, visual_state_embedding, single_images=True, COT=False, *args, **kwargs):
        super().__init__('Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4', instruction_prompt_file_path, max_history, visual_state_embedding, single_images, COT)
            
