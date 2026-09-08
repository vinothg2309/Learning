import re
import docx
import os

def clean_tree_artifacts(text):
    """
    Converts visual tree nodes (├─, └─) into clean Markdown bullet items
    while maintaining appropriate nesting levels.
    """
    text = text.replace("│  ├─", "    - ")
    text = text.replace("│  └─", "    - ")
    text = text.replace("├─", "  - ")
    text = text.replace("└─", "  - ")
    text = text.replace("│", "")
    return text

def convert_docx_to_md(docx_path, md_path):
    """
    Parses a single DOCX file, applies structure logic, and 
    saves the output cleanly to the destination Markdown path.
    """
    doc = docx.Document(docx_path)
    md_lines = []
    
    in_code_block = False
    in_diagram = False
    
    for paragraph in doc.paragraphs:
        text = paragraph.text
        style_name = paragraph.style.name.lower()
        
        # 1. Detect and preserve ASCII architecture diagrams
        if text.startswith("┌───") or text.startswith("└───") or "──────" in text:
            if not in_diagram:
                md_lines.append("```text")
                in_diagram = True
            md_lines.append(text)
            continue
        elif in_diagram and not (text.startswith("┌") or text.startswith("│") or text.startswith("└") or "──" in text):
            md_lines.append("```\n")
            in_diagram = False
            
        # 2. Detect and encapsulate code blocks or workflow structures
        if "@dag" in text or "def workflow" in text or "if evidence_quality" in text:
            if not in_code_block:
                md_lines.append("```python")
                in_code_block = True
            md_lines.append(text)
            continue
        elif in_code_block and (text.strip() == "" or text.startswith("---") or "Questions" in text):
            md_lines.append("```\n")
            in_code_block = False

        # 3. Structural Header Mapping
        if style_name.startswith("heading 1"):
            md_lines.append(f"\n# {text.strip()}\n")
        elif style_name.startswith("heading 2") or re.match(r"^Q\d+:", text.strip()):
            md_lines.append(f"\n## {text.strip()}\n")
        elif style_name.startswith("heading 3") or text.strip().startswith("Step ") or text.strip().startswith("Stage "):
            md_lines.append(f"\n### {text.strip()}\n")
        elif text.strip() == "---":
            md_lines.append("\n---\n")
        else:
            # Clean structure drawing artifacts if text is outside literal blocks
            if not in_code_block and not in_diagram:
                cleaned_text = clean_tree_artifacts(text)
                md_lines.append(cleaned_text)
            else:
                md_lines.append(text)
                
    # Safeguard close for block scopes at EOF
    if in_code_block or in_diagram:
        md_lines.append("```")

    # Write processed code out using UTF-8 encoding
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

def batch_process_folders(source_folder, dest_folder):
    """
    Iterates through all valid .docx files inside the source folder
    and exports formatted markdown files to the destination folder.
    """
    # Verify presence of input source
    if not os.path.exists(source_folder):
        print(f"Error: Source directory '{source_folder}' does not exist.")
        return
        
    # Automatically initialize target directory if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created destination directory: {dest_folder}")
        
    converted_count = 0
    
    for file_name in os.listdir(source_folder):
        # Exclude temporary Word operational files (~$)
        if file_name.endswith(".docx") and not file_name.startswith("~$"):
            docx_path = os.path.join(source_folder, file_name)
            
            md_file_name = file_name.replace(".docx", ".md")
            md_path = os.path.join(dest_folder, md_file_name)
            
            try:
                convert_docx_to_md(docx_path, md_path)
                print(f"✓ Converted: {file_name} -> {os.path.join(dest_folder, md_file_name)}")
                converted_count += 1
            except Exception as e:
                print(f"✗ Failed to convert {file_name}: {str(e)}")
                
    print(f"\nBatch processing complete. Total files successfully processed: {converted_count}")

if __name__ == "__main__":
    # Define your folder configurations here
    SOURCE_DIR = "NeMo\\NIM"       # Folder containing your .docx files
    DESTINATION_DIR = "NeMo\\NIM"  # Target output directory for your .md files
    
    batch_process_folders(SOURCE_DIR, DESTINATION_DIR)