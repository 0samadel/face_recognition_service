# --- Python script: image_to_base64.py ---
import base64
import sys

def image_to_base64_string(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_to_base64.py <path_to_image_file>")
    else:
        image_path = sys.argv[1]
        base64_str = image_to_base64_string(image_path)
        if base64_str:
            print("\nBase64 Encoded String:")
            print(base64_str)
            # You can also save it to a file if it's too long for the console
            # with open("output_base64.txt", "w") as f:
            #     f.write(base64_str)
            # print("\nSaved base64 string to output_base64.txt")