import os
import nbformat as nbf
import docx

def is_paragraph_styled_as_code(p):
    """
    Optional helper to identify if a paragraph uses an explicit 
    code/monospace typography layout in Microsoft Word.
    """
    style_name = p.style.name.lower()
    if "code" in style_name or "preformatted" in style_name or "markup" in style_name:
        return True
    # Check if the run uses Courier New or Consolas monospace fonts
    for run in p.runs:
        if run.font.name in ["Courier New", "Consolas", "Lucida Console"]:
            return True
    return False

def parse_docx_to_cells(docx_path):
    """
    Parses a Word document dynamically tracking raw strings, 
    ASCII diagrams, and code boundaries using state machine triggers.
    """
    doc = docx.Document(docx_path)
    cells = []
    
    current_block = []
    in_code_mode = False
    in_diagram_mode = False

    for paragraph in doc.paragraphs:
        text = paragraph.text
        stripped = text.strip()

        # --- TRIGGER 1: CODE FENCE DETECTIONS (```python, ```bash, etc.) ---
        if stripped.startswith("```"):
            if in_code_mode:
                # Closing an active code block -> Flush it as a Jupyter Code Cell
                if current_block:
                    cells.append(nbf.v4.new_code_cell("\n".join(current_block)))
                current_block = []
                in_code_mode = False
            elif in_diagram_mode:
                # Closing an active diagram block -> Flush it as a Markdown block
                if current_block:
                    diagram_md = f"```text\n" + "\n".join(current_block) + "\n```"
                    cells.append(nbf.v4.new_markdown_cell(diagram_md))
                current_block = []
                in_diagram_mode = False
            else:
                # Opening a brand new block context
                # Check if it specifies text/diagram format or executable code syntax
                if "text" in stripped or "diagram" in stripped:
                    in_diagram_mode = True
                else:
                    in_code_mode = True
            continue

        # --- TRIGGER 2: BOX-DRAWING DIAGRAMS (┌───, ├───) ---
        if stripped.startswith("┌──") or stripped.startswith("└──") or "──────" in stripped:
            if not in_diagram_mode and not in_code_mode:
                # Flush previous markdown elements if any exist
                if current_block:
                    cells.append(nbf.v4.new_markdown_cell("\n\n".join(current_block)))
                    current_block = []
                in_diagram_mode = True
            current_block.append(text)
            continue

        # --- STREAM PROCESSING BASED ON CAPTURED STATE ---
        if in_code_mode or in_diagram_mode:
            # We are currently inside a multi-line literal chunk, keep collecting rows
            current_block.append(text)
        else:
            # Fallback heuristic: If no markdown backticks exist, look for distinct 
            # software pattern signatures (e.g. pip commands, imports, comments followed by statements)
            is_implicit_code = (
                stripped.startswith("pip install") or 
                stripped.startswith("import ") or 
                stripped.startswith("from ") or
                stripped.startswith("@") or
                stripped.startswith("def ") or
                stripped.startswith("async def ") or
                is_paragraph_styled_as_code(paragraph)
            )

            if is_implicit_code:
                # If we were collecting normal markdown text paragraphs, flush them first
                if current_block:
                    cells.append(nbf.v4.new_markdown_cell("\n\n".join(current_block)))
                    current_block = []
                # Immediately create a clean standalone Code Cell
                cells.append(nbf.v4.new_code_cell(stripped))
            else:
                # Collect standard paragraph lines or markdown headers
                if stripped:
                    current_block.append(text)
                elif current_block:
                    # Treat double newlines as cell break parameters
                    cells.append(nbf.v4.new_markdown_cell("\n\n".join(current_block)))
                    current_block = []

    # --- CLEANUP AT END OF FILE (EOF) ---
    if current_block:
        if in_code_mode:
            cells.append(nbf.v4.new_code_cell("\n".join(current_block)))
        elif in_diagram_mode:
            diagram_md = f"```text\n" + "\n".join(current_block) + "\n```"
            cells.append(nbf.v4.new_markdown_cell(diagram_md))
        else:
            cells.append(nbf.v4.new_markdown_cell("\n\n".join(current_block)))

    return cells

def batch_convert_to_notebooks(source_folder, destination_folder):
    """
    Main controller to verify folders, map files, and execute compilation safely.
    """
    source_folder = os.path.normpath(source_folder)
    destination_folder = os.path.normpath(destination_folder)

    if not os.path.exists(source_folder):
        print(f"Error: Source directory missing: {source_folder}")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    success_count = 0
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".docx") and not file_name.startswith("~$"):
            source_path = os.path.join(source_folder, file_name)
            
            # Setup output filename
            notebook_name = os.path.splitext(file_name)[0] + ".ipynb"
            dest_path = os.path.join(destination_folder, notebook_name)
            
            try:
                notebook_obj = nbf.v4.new_notebook()
                notebook_obj['cells'] = parse_docx_to_cells(source_path)
                
                with open(dest_path, "w", encoding="utf-8") as f:
                    nbf.write(notebook_obj, f)
                print(f"✓ Successfully Generated: {notebook_name}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed processing file {file_name}. Error: {str(e)}")

    print(f"\nCompleted! Generated {success_count} fully formatted Jupyter Notebooks.")


if __name__ == "__main__":
    # Configurable parameters (Raw strings rule out Windows backslash encoding bugs)
    # SOURCE_DIR = r".\Learning\AGENTS\LANGCHAIN_GRAPH_SMITH\docs"
    # DEST_DIR = r".Learning\AGENTS\LANGCHAIN_GRAPH_SMITH"

    SOURCE_DIR = "python\doc"       # Folder containing your .docx files
    DEST_DIR = "python" 
    
    batch_convert_to_notebooks(SOURCE_DIR, DEST_DIR)