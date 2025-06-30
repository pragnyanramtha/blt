# utils.py
import requests
import re

def download_gutenberg_text(url):
    """
    Downloads text from a Project Gutenberg URL and cleans the header/footer.
    """
    print(f"Downloading text from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    text = response.text

    # Use regex to find the start and end of the main content
    start_match = re.search(r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*', text)
    end_match = re.search(r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*', text)

    if start_match and end_match:
        start_pos = start_match.end()
        end_pos = end_match.start()
        print("Successfully stripped Gutenberg header and footer.")
        return text[start_pos:end_pos].strip()
    else:
        print("Warning: Could not find standard Gutenberg header/footer. Returning full text.")
        return text