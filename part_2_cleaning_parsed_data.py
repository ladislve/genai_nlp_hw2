import os
import re

def fix_gibberish_segment(match): # wrong encoding
    text_segment = match.group(0) # takes substring
    decoded = text_segment.encode('cp1252').decode('cp1251') # takes as win1252 converts into win1251
    return decoded 

def fix_mixed_line(text): # searches in range, applies function above
    text = re.sub(r'[\u00C0-\u00FF]+', fix_gibberish_segment, text)
    return text

def fix_font_offset(text): # fixes wrong unicode code points caused by bad font mapping
    chars = []
    for char in text:
        code = ord(char)
        if 0x0180 <= code <= 0x02AF:
            shifted_code = code + 470
            if 0x0400 <= shifted_code <= 0x04FF:
                chars.append(chr(shifted_code))
            else:
                chars.append(char)
        else:
            chars.append(char)
    return "".join(chars)

def clean_artifacts(text): # removes some visual clutter left behind
    text = text.replace('\U00033FE0', '') 
    text = re.sub(r'[-_—]{3,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\.{4,}', ' ... ', text)
    text = re.sub(r'^\W+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_file(file_path): # performs cleaning for one file
    print(f"cleaning {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        line = fix_font_offset(line)
        line = fix_mixed_line(line)
        cleaned_lines.append(line)
    
    full_text = "".join(cleaned_lines)
    final_text = clean_artifacts(full_text)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

def main(): # calls cleaning for one file per time
    output_base = "parsed_output"
    if not os.path.exists(output_base): return

    for root, dirs, files in os.walk(output_base):
        for file in files:
            if file.endswith(".md"):
                process_file(os.path.join(root, file))

    print("cleaning complete")

if __name__ == "__main__":
    main()