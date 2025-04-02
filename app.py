from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import io
import base64
import time
import logging
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Gemini with the model of your choice
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not set. Please configure it in .env file.")
    llm = None
else:
    try:
        llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Change from "gemini-pro-vision" to "gemini-1.5-flash"
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    max_output_tokens=2048,
    convert_system_message_to_human=True
)

    except Exception as e:
        logger.error(f"Error initializing Gemini: {str(e)}")
        llm = None

def optimize_image(image):
    """Optimize image size and quality for faster processing"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.thumbnail((800, 800), Image.LANCZOS)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return buffered.getvalue()
    except Exception as e:
        logger.error(f"Error optimizing image: {str(e)}")
        raise

def process_image_with_gemini(image_data, prompt):
    try:
        logger.info("Processing image with Gemini AI")
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        system_prompt = """You are a helpful assistant that analyzes food products and provides detailed nutritional information.
        You must ALWAYS respond with ONLY valid JSON, no additional text or explanation.
        
        For food products, respond with this exact format:
        {
            "isFoodProduct": true,
            "data": {
                "id": "unique_id",
                "name": "product_name",
                "image": "base64_image_data",
                "brandOwner": "brand_owner",
                "brandName": "brand_name",
                "ingredients": "list_of_ingredients",
                "servingSize": 100,
                "servingSizeUnit": "g",
                "packageWeight": "100g",
                "additives": ["additive1", "additive2"],
                "nutrients": [
                    {
                        "id": 1,
                        "name": "Calories",
                        "code": "ENERC_KCAL",
                        "amount": 100,
                        "unitName": "kcal",
                        "rate": 5,
                        "ratedIndex": 1,
                        "metric": {
                            "name": "Energy",
                            "img": "energy.png",
                            "messages": ["Low energy", "High energy"],
                            "benchmarks_100g": [50, 200],
                            "benchmarks_unit": "kcal",
                            "benchmarks_ratio": [0.25, 1],
                            "rates": [1, 5]
                        }
                    }
                ]
            }
        }
        
        For non-food products, respond with this exact format:
        {
            "isFoodProduct": false,
            "error": "This image does not appear to be a food product",
            "description": "Brief description of what the image actually shows"
        }
        
        Remember: Respond with ONLY the JSON object, no additional text or explanation."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Analyze this image and provide information in JSON format. If it's a food product, include nutritional details. If not, indicate it's not a food product."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ])
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Log the raw response for debugging
        logger.info(f"Raw response from Gemini: {response_text}")
        
        # Try to parse the response as JSON
        try:
            import json
            response_data = json.loads(response_text)
            return json.dumps(response_data)  # Return stringified JSON
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            # If JSON parsing fails, try to extract JSON from the response
            try:
                # Look for JSON-like content between curly braces
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    response_data = json.loads(json_str)
                    return json.dumps(response_data)
            except Exception as e:
                logger.error(f"Failed to extract JSON from response: {str(e)}")
            
            # If all parsing attempts fail, return a structured error response
            return json.dumps({
                "isFoodProduct": False,
                "error": "Failed to parse response as JSON",
                "description": response_text
            })
            
    except Exception as e:
        logger.error(f"Error processing image with Gemini: {str(e)}")
        return json.dumps({
            "isFoodProduct": False,
            "error": "Error processing image",
            "description": str(e)
        })

@app.route('/process-image', methods=['POST'])
def process_image():
    if llm is None:
        return jsonify({'success': False, 'error': 'Gemini LLM not initialized'}), 500
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        image = Image.open(image_file)
        optimized_image_data = optimize_image(image)
        response_text = process_image_with_gemini(optimized_image_data, "Analyze this image and provide information in JSON format.")
        
        # Parse the response as JSON
        import json
        try:
            response_data = json.loads(response_text)
            return jsonify(response_data)
        except json.JSONDecodeError as e:
            return jsonify({
                'success': False,
                'error': 'Invalid JSON response from Gemini',
                'details': str(e),
                'raw_response': response_text
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
